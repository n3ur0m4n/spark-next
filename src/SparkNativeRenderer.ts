import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import { ExtSplats } from "./ExtSplats";
import { PackedSplats } from "./PackedSplats";
import { PagedSplats } from "./PagedSplats";
import type { SparkHostRenderer } from "./RendererAdapter";
import type { SparkRenderStats } from "./SparkRenderStats";
import type { SparkRenderer } from "./SparkRenderer";
import { SplatAccumulator } from "./SplatAccumulator";
import { SplatEdit } from "./SplatEdit";
import { SplatGenerator } from "./SplatGenerator";
import { SplatMesh } from "./SplatMesh";
import { SplatWorker } from "./SplatWorker";
import { WebGPUComposite } from "./WebGPUComposite";
import type { NativeMeshGeneration } from "./WebGPUGeneration";
import {
  type WebGPUPickPixels,
  WebGPUSplatBackend,
} from "./WebGPUSplatBackend";

// Camera motion below this (world units, or radians for directional sorts)
// keeps the last ordering, like PlayCanvas.
const SORT_EPSILON = 0.001;
// Rotation slack between the sort cone and the view's half-diagonal.
const CONE_MARGIN = (2 * Math.PI) / 180;

/**
 * True when the sort cone of the last sort, rotated by `angle`, still
 * contains the whole perspective view, so its culling leaves no holes.
 */
function coneCoversView(
  camera: THREE.Camera,
  sortCullCos: number,
  angle: number,
  sortCullFov: number,
): boolean {
  const perspective = camera as THREE.PerspectiveCamera;
  if (sortCullCos === -2 || !perspective.isPerspectiveCamera) return false;
  if (perspective.view?.enabled) return false;
  const tanV =
    Math.tan((perspective.fov * Math.PI) / 360) / (perspective.zoom || 1);
  const halfDiagonal = Math.atan(tanV * Math.hypot(1, perspective.aspect));
  const coneHalf = (Math.min(sortCullFov, 360) * Math.PI) / 360;
  return angle + halfDiagonal + CONE_MARGIN <= coneHalf;
}

/** The native submission path; the WebGL renderer and LoD traversal stay unchanged. */
export class SparkNativeRenderer {
  readonly backend: WebGPUSplatBackend;
  private composite: WebGPUComposite | null = null;
  private targetComposite: WebGPUComposite | null = null;
  private queue: Promise<void> = Promise.resolve();
  private disposed = false;
  private renderVisible = true;
  private readonly hostSize = new THREE.Vector2();
  private readonly hostViewport = new THREE.Vector4();
  private sequence = 0;
  private currentSignature = "";
  /** Mapping (signature without the view) of the last radix-sorted ordering. */
  private sortedMapping = "";
  private lastSortTime = Number.NEGATIVE_INFINITY;
  /** View of the last radix sort (sort epsilon and sort-cone reuse). */
  private readonly lastSortOrigin = new THREE.Vector3(Number.NaN, 0, 0);
  private readonly lastSortDir = new THREE.Vector3();
  /** The committed ordering was sorted for an earlier view. */
  private orderingStale = false;
  private staleTimeout: ReturnType<typeof setTimeout> | undefined;
  private updateQueue: Promise<SplatMesh[]> = Promise.resolve([]);
  private worker: SplatWorker | null = null;
  private ordering = new Uint32Array(0);
  private readonly identities = new WeakMap<object, number>();
  private nextIdentity = 1;
  private stats: SparkRenderStats = {
    generation: 0,
    selectedSplats: 0,
    drawnSplats: 0,
    ordering: "readback",
  };

  getRenderStats(): SparkRenderStats {
    return {
      ...this.stats,
      drawnSplats: this.renderVisible ? this.stats.drawnSplats : 0,
    };
  }

  private isVisible(camera: THREE.Camera): boolean {
    for (
      let node: THREE.Object3D | null = this.owner;
      node;
      node = node.parent
    ) {
      if (!node.visible) return false;
    }
    return camera.layers.test(this.owner.layers);
  }

  async readRenderStatsAsync(): Promise<SparkRenderStats> {
    if (this.disposed) throw new Error("SparkRenderer is disposed");
    const snapshot = this.getRenderStats();
    if (snapshot.drawnSplats !== null) return snapshot;
    // The copy is submitted synchronously, before another generation can reuse
    // this slot. A late result describes its original generation only.
    const drawnSplats = await this.backend.readNativeDrawCount();
    const measured = { ...snapshot, drawnSplats };
    if (!this.disposed && this.stats.generation === snapshot.generation) {
      this.stats = measured;
      this.owner.activeSplats = drawnSplats;
    }
    return measured;
  }

  private identity(value?: object): number {
    if (!value) return 0;
    let id = this.identities.get(value);
    if (id === undefined) {
      id = this.nextIdentity++;
      this.identities.set(value, id);
    }
    return id;
  }

  constructor(
    private readonly owner: SparkRenderer<SparkHostRenderer>,
    private readonly renderer: WebGPURenderer,
    device: GPUDevice,
  ) {
    this.backend = new WebGPUSplatBackend(device);
  }

  assertSupported(scene: THREE.Scene, camera: THREE.Camera) {
    const requireWebGL = (feature: string): never => {
      throw new Error(
        `Spark: ${feature} requires THREE.WebGLRenderer; native output was not submitted.`,
      );
    };
    if (this.disposed) throw new Error("SparkRenderer is disposed");
    if (camera instanceof THREE.ArrayCamera) requireWebGL("array cameras");
    if (this.backend.deviceLost)
      throw new Error(
        `Spark: host GPU device lost: ${this.backend.deviceLostReason}`,
      );
    if (this.backend.uncapturedErrorCount)
      throw new Error(
        `Spark: native GPU validation failed: ${this.backend.firstUncapturedError}`,
      );
    if (this.renderer.getRenderTarget())
      requireWebGL("specialized render targets");
    if (this.renderer.logarithmicDepthBuffer) requireWebGL("logarithmic depth");
    if (
      camera.reversedDepth ||
      ("reversedDepthBuffer" in this.renderer &&
        this.renderer.reversedDepthBuffer === true)
    )
      requireWebGL("reversed depth");
    if (this.renderer.getScissorTest()) requireWebGL("scissor test");
    if (
      !this.renderer.autoClear ||
      !this.renderer.autoClearColor ||
      !this.renderer.autoClearDepth
    )
      requireWebGL("manual clearing");
    // Both methods use logical pixels, including on high-DPI canvases.
    this.renderer.getSize(this.hostSize);
    this.renderer.getViewport(this.hostViewport);
    if (
      this.hostViewport.x !== 0 ||
      this.hostViewport.y !== 0 ||
      this.hostViewport.z !== this.hostSize.x ||
      this.hostViewport.w !== this.hostSize.y
    )
      requireWebGL("custom viewports");
    if (this.renderer.xr.isPresenting) requireWebGL("WebXR");
    if (this.renderer.samples > 0) requireWebGL("native MSAA");
    if (scene.overrideMaterial) requireWebGL("scene override materials");
    if (this.owner.covSplats) requireWebGL("covariance splats");
    if (
      !(
        camera instanceof THREE.PerspectiveCamera ||
        camera instanceof THREE.OrthographicCamera
      )
    ) {
      requireWebGL("custom cameras");
    }
    scene.updateMatrixWorld();
    scene.traverseVisible((node) => {
      if (!camera.layers.test(node.layers)) return;
      if (
        node instanceof THREE.Mesh ||
        node instanceof THREE.Line ||
        node instanceof THREE.Points ||
        node instanceof THREE.Sprite
      ) {
        const materials = Array.isArray(node.material)
          ? node.material
          : [node.material];
        // Transparent objects are drawn by WebGPUComposite around the splat
        // pass by renderOrder; transmission still needs the WebGL path.
        if (
          materials.some(
            (material) =>
              "transmission" in material &&
              typeof material.transmission === "number" &&
              material.transmission > 0,
          )
        ) {
          requireWebGL("transmissive Three.js objects");
        }
      }
      if (!(node instanceof SplatGenerator)) return;
      if (!(node instanceof SplatMesh) || !node.hasNativeSourceGenerator())
        requireWebGL("custom splat generators");
      if (!(node instanceof SplatMesh)) return;
      if (node.covSplats) requireWebGL("covariance splats");
      if (
        node.objectModifiers?.length ||
        node.worldModifiers?.length ||
        node.covObjectModifiers?.length ||
        node.covWorldModifiers?.length
      )
        requireWebGL("modifiers");
      if (node.skinning) requireWebGL("skinning");
      if (node.splatRgba) requireWebGL("baked splat RGBA");
      if (
        node.enableViewToObject ||
        node.enableViewToWorld ||
        node.enableWorldToView ||
        node.showLodPage != null
      )
        requireWebGL("custom view-dependent generation");
      const e = node.matrixWorld.elements;
      const x = new THREE.Vector3(e[0], e[1], e[2]);
      const y = new THREE.Vector3(e[4], e[5], e[6]);
      const z = new THREE.Vector3(e[8], e[9], e[10]);
      const scale = x.lengthSq();
      const tolerance = 1e-5 * Math.max(scale, 1e-20);
      if (
        scale === 0 ||
        node.matrixWorld.determinant() <= 0 ||
        Math.abs(y.lengthSq() - scale) > tolerance ||
        Math.abs(z.lengthSq() - scale) > tolerance ||
        Math.abs(x.dot(y)) > tolerance ||
        Math.abs(x.dot(z)) > tolerance ||
        Math.abs(y.dot(z)) > tolerance
      )
        requireWebGL("nonuniform, mirrored or sheared splat transforms");
    });
  }

  /**
   * Each call keeps its camera snapshot and has its own submission promise.
   * `target` receives the full composite instead of the canvas: the canvas
   * image (same tone mapping) as linear premultiplied color. It has its own
   * intermediate target so captures do not resize the canvas one.
   */
  renderAsync(
    scene: THREE.Scene,
    camera: THREE.Camera,
    { target = null }: { target?: THREE.RenderTarget | null } = {},
  ): Promise<void> {
    if (this.disposed)
      return Promise.reject(new Error("SparkRenderer is disposed"));
    // ArrayCamera inherits PerspectiveCamera, but needs one generation per view.
    // Reject it before cloning can discard its subcamera list (Three r180).
    if (camera instanceof THREE.ArrayCamera)
      return Promise.reject(
        new Error(
          "Spark: array cameras require THREE.WebGLRenderer; native output was not submitted.",
        ),
      );
    // Camera.copy in Three r180 does not retain the reversed-depth flag.
    if (camera.reversedDepth)
      return Promise.reject(
        new Error(
          "Spark: reversed depth requires THREE.WebGLRenderer; native output was not submitted.",
        ),
      );
    camera.updateWorldMatrix(true, false);
    const snapshot = camera.clone(false);
    snapshot.matrix.copy(camera.matrixWorld);
    snapshot.matrixWorld.copy(camera.matrixWorld);
    snapshot.matrixWorldInverse.copy(camera.matrixWorldInverse);
    snapshot.matrixAutoUpdate = false;
    snapshot.matrixWorldAutoUpdate = false;
    const task = this.queue
      .catch(() => {})
      .then(async () => {
        this.assertSupported(scene, snapshot);
        const visible = this.isVisible(snapshot);
        const readSize = () => {
          if (target) this.owner.renderSize.set(target.width, target.height);
          else this.renderer.getDrawingBufferSize(this.owner.renderSize);
        };
        readSize();
        if (visible)
          await this.owner.onBeforeRender(this.renderer, scene, snapshot);
        this.assertSupported(scene, snapshot);
        const key = target ? "targetComposite" : "composite";
        let composite = this[key];
        if (!composite) {
          composite = await WebGPUComposite.create();
          if (this.disposed) {
            composite.dispose();
            throw new Error("SparkRenderer is disposed");
          }
          this[key] = composite;
        }
        // The host can resize while generation/readback or module loading is
        // pending. Projection uniforms must match the attachments drawn below.
        readSize();
        this.renderVisible = visible && this.isVisible(snapshot);
        this.owner.updateStochastic(snapshot);
        this.backend.stochastic = this.owner.stochasticActive;
        this.updateUniforms(snapshot);
        composite.render(
          this.renderer,
          scene,
          snapshot,
          ({ color, depth, srgb }) => {
            if (!this.renderVisible) return;
            this.backend.premultipliedAlpha = this.owner.premultipliedAlpha;
            // "bytes": blend through a non-sRGB alias of the sRGB target.
            const bytes = srgb === "bytes" && color.format.endsWith("-srgb");
            this.backend.ensurePipeline(
              bytes
                ? (color.format.slice(0, -5) as GPUTextureFormat)
                : color.format,
            );
            if (
              !this.backend.setExternalDepthTexture(
                depth,
                color.width,
                color.height,
              )
            ) {
              throw new Error(
                "Spark: incompatible native composite depth attachment",
              );
            }
            this.backend.renderSeparatePass(
              color,
              this.owner.activeSplats,
              true,
              bytes,
            );
          },
          this.owner.renderOrder,
          target,
          this.srgbBlendActive(),
        );
      });
    this.queue = task;
    return task;
  }

  /**
   * pickAsync() render: the w x h CSS-pixel crop at (ox, oy) of the
   * fullW x fullH view, drawn from the last submitted frame's ordering. Queued
   * after pending renders so it sees the ordering they committed.
   */
  pickPixels(
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
    crop: {
      fullW: number;
      fullH: number;
      ox: number;
      oy: number;
      w: number;
      h: number;
      alphaThreshold: number;
      farBig: number;
      solid: boolean;
    },
  ): Promise<WebGPUPickPixels> {
    if (this.disposed)
      return Promise.reject(new Error("SparkRenderer is disposed"));
    camera.updateWorldMatrix(true, false);
    const pickCamera = camera.clone();
    pickCamera.setViewOffset(
      crop.fullW,
      crop.fullH,
      crop.ox,
      crop.oy,
      crop.w,
      crop.h,
    );
    let result: Promise<WebGPUPickPixels> | undefined;
    const submitted = this.queue.catch(() => {}).then(() => {
      if (this.disposed) throw new Error("SparkRenderer is disposed");
      this.updateUniforms(pickCamera, { width: crop.w, height: crop.h });
      // readPickPixels è sincrono fino a device.queue.submit: la coda si libera col submit, il readback resta fuori catena
      result = this.backend.readPickPixels({
        width: crop.w,
        height: crop.h,
        numInstances: this.renderVisible ? this.owner.activeSplats : 0,
        alphaThreshold: crop.alphaThreshold,
        farBig: crop.farBig,
        solid: crop.solid,
      });
    });
    this.queue = submitted;
    return submitted.then(() => result!);
  }

  /** sRGB splat blending (WebGPUComposite srgbBlend): needs an sRGB output. */
  private srgbBlendActive(): boolean {
    return (
      this.owner.srgbBlend &&
      this.renderer.outputColorSpace === THREE.SRGBColorSpace
    );
  }

  private updateUniforms(
    camera: THREE.Camera,
    pick?: { width: number; height: number },
  ) {
    const owner = this.owner;
    const uniforms = owner.uniforms;
    const typed = camera as THREE.PerspectiveCamera | THREE.OrthographicCamera;
    typed.coordinateSystem = THREE.WebGPUCoordinateSystem;
    typed.updateProjectionMatrix();
    if (pick) uniforms.renderSize.value.set(pick.width, pick.height);
    else uniforms.renderSize.value.copy(owner.renderSize);
    uniforms.near.value = typed.near;
    uniforms.far.value = typed.far;
    const transform = camera.matrixWorld.clone().invert();
    if (!owner.display.extSplats)
      transform.multiply(
        new THREE.Matrix4().makeTranslation(owner.display.viewOrigin),
      );
    transform.decompose(
      uniforms.renderToViewPos.value,
      uniforms.renderToViewQuat.value,
      new THREE.Vector3(),
    );
    uniforms.renderToViewBasis.value.setFromMatrix4(transform);
    for (const key of [
      "maxStdDev",
      "minPixelRadius",
      "maxPixelRadius",
      "minAlpha",
      "preBlurAmount",
      "blurAmount",
      "focalDistance",
      "apertureAngle",
      "falloff",
      "clipXY",
      "focalAdjustment",
    ] as const) {
      uniforms[key].value = owner[key];
    }
    uniforms.enable2DGS.value = owner.enable2DGS;
    uniforms.lodInflate.value = owner.lodInflate;
    // sRGB blending writes sRGB-encoded colors (see renderSeparatePass).
    uniforms.encodeLinear.value = !this.srgbBlendActive();
    uniforms.enableExtSplats.value = owner.display.extSplats;
    uniforms.enableCovSplats.value = false;
    uniforms.time.value = owner.display.time;
    uniforms.deltaTime.value = owner.display.deltaTime;
    // A splat whose quad covers the pick window can have its center outside
    // the tiny crop: no center-based XY clipping while picking.
    if (pick) uniforms.clipXY.value = 1e6;
    this.backend.updateUniforms(
      uniforms,
      typed.projectionMatrix,
      typed instanceof THREE.OrthographicCamera,
      pick ? this.backend.pickUniformBuffers() : undefined,
    );
  }

  /** Runs ordinary mesh callbacks, then snapshots and commits one complete generation. */
  update(scene: THREE.Scene, camera: THREE.Camera): Promise<SplatMesh[]> {
    const task = this.updateQueue
      .catch(() => [])
      .then(() => this.updateFrame(scene, camera));
    this.updateQueue = task;
    return task;
  }

  private async updateFrame(
    scene: THREE.Scene,
    camera: THREE.Camera,
  ): Promise<SplatMesh[]> {
    this.assertSupported(scene, camera);
    camera.updateWorldMatrix(true, false);
    const owner = this.owner;
    const sortRadial = owner.sortRadial;
    const sortMode = owner.webgpuSort;
    // Same cone as SparkRenderer.updateInternal (sortCullFov).
    const sortCullCos =
      owner.sortCullFov > 0 &&
      !(camera as THREE.OrthographicCamera).isOrthographicCamera
        ? Math.cos((Math.min(owner.sortCullFov, 360) * Math.PI) / 360)
        : -2;
    const next = new SplatAccumulator({
      extSplats: owner.accumExtSplats,
      covSplats: false,
    });
    next.time = owner.timer.getElapsed();
    next.deltaTime = owner.timer.getDelta();
    next.viewToWorld.copy(camera.matrixWorld);
    camera.getWorldPosition(next.viewOrigin);
    camera.getWorldDirection(next.viewDirection);
    // Same global edit set as SplatAccumulator: visible SplatEdits outside
    // any SplatMesh apply to every editable mesh.
    const globalEdits = new Set<SplatEdit>();
    scene.traverseVisible((node) => {
      if (!(node instanceof SplatEdit)) return;
      let ancestor = node.parent;
      while (ancestor && !(ancestor instanceof SplatMesh))
        ancestor = ancestor.parent;
      if (!ancestor) globalEdits.add(node);
    });
    const all: SplatMesh[] = [];
    scene.traverse((node) => {
      if (node instanceof SplatMesh && camera.layers.test(node.layers))
        all.push(node);
    });
    for (const node of all) {
      node.update({
        object: node,
        time: next.time,
        deltaTime: next.deltaTime,
        viewToWorld: next.viewToWorld,
        camera,
        renderSize: owner.renderSize,
        globalEdits: Array.from(globalEdits),
        lodIndices: owner.enableLod ? owner.lodInstances.get(node) : undefined,
      });
    }
    this.assertSupported(scene, camera);
    const visible: SplatMesh[] = [];
    scene.traverseVisible((node) => {
      if (node instanceof SplatMesh && camera.layers.test(node.layers))
        visible.push(node);
    });
    const counts = visible.map((node) => {
      const pending =
        owner.enableLod &&
        owner.enableDriveLod &&
        node.enableLod !== false &&
        (node.packedSplats?.lodSplats || node.extSplats?.lodSplats) &&
        !owner.lodInstances.has(node);
      return pending ? 0 : node.numSplats;
    });
    const layout = next.generateMapping(counts);
    const meshes: NativeMeshGeneration[] = [];
    for (let i = 0; i < visible.length; i++) {
      const node = visible[i];
      const { base, count } = layout.mapping[i];
      if (!count) continue;
      const source = node.context.splats;
      if (source instanceof PagedSplats) {
        meshes.push(this.pagedGeneration(node, source, base, count));
        next.mapping.push({
          node,
          base,
          count,
          version: node.version,
          mappingVersion: node.mappingVersion,
          generator: node.generator,
        });
        next.numSplats = Math.max(next.numSplats, base + count);
        continue;
      }
      if (!(source instanceof PackedSplats || source instanceof ExtSplats))
        throw new Error(
          "Spark: native source requires PackedSplats or ExtSplats",
        );
      if (source instanceof PackedSplats && source.target) {
        throw new Error(
          "Spark: GPU-generated source textures require THREE.WebGLRenderer",
        );
      }
      const extended = source instanceof ExtSplats;
      const primary = extended ? source.extArrays[0] : source.packedArray;
      const secondary = extended ? source.extArrays[1] : undefined;
      const indices =
        owner.enableLod && node.context.enableLod.value
          ? owner.lodInstances.get(node)?.indices
          : undefined;
      if (
        !primary ||
        primary.length < source.numSplats * 4 ||
        (extended && (!secondary || secondary.length < source.numSplats * 4)) ||
        (indices && indices.length < count)
      )
        throw new Error("Spark: incomplete native source or selection");
      const enc = extended ? undefined : source.splatEncoding;
      const sh = [
        source.extra.sh1,
        source.extra.sh2,
        extended ? source.extra.sh3a : source.extra.sh3,
        extended ? source.extra.sh3b : undefined,
      ].map((value) => (value instanceof Uint32Array ? value : undefined));
      const numSh = Math.min(node.maxSh, source.maxSh, source.getNumSh());
      for (let band = 0; band < (numSh === 3 && extended ? 4 : numSh); band++) {
        const words = source.numSplats * (!extended && band === 0 ? 2 : 4);
        const coefficients = sh[band];
        if (!coefficients || coefficients.length < words) {
          throw new Error("Spark: incomplete native spherical-harmonic source");
        }
      }
      const transform = node.context.transform;
      meshes.push({
        source,
        revision: source.dataVersion,
        primary,
        secondary,
        sh,
        sourceCount: source.numSplats,
        sourceExt: extended,
        base,
        count,
        indices,
        rotation: transform.rotate.value.toArray(),
        translationScale: [
          ...transform.translate.value.toArray(),
          transform.scale.value,
        ],
        localViewOrigin: node.context.viewToObject.translate.value.toArray(),
        encoding: [
          enc?.rgbMin ?? 0,
          enc?.rgbMax ?? 1,
          enc?.lnScaleMin ?? -12,
          enc?.lnScaleMax ?? 9,
        ],
        shMaxOpacity: [
          enc?.sh1Max ?? 1,
          enc?.sh2Max ?? 1,
          enc?.sh3Max ?? 1,
          enc?.lodOpacity ? 2 : 1,
        ],
        numSh,
        recolor: node.context.recolor.value.toArray(),
        introDots: node.introDots.active ? node.introDots.time : -1,
        edits: this.meshEdits(node),
      });
      next.mapping.push({
        node,
        base,
        count,
        version: node.version,
        mappingVersion: node.mappingVersion,
        generator: node.generator,
      });
      next.numSplats = Math.max(next.numSplats, base + count);
    }
    const view = JSON.stringify([
      next.viewToWorld.toArray(),
      next.viewOrigin.toArray(),
      next.viewDirection.toArray(),
    ]);
    const mapping = JSON.stringify([
      sortMode,
      sortRadial,
      sortCullCos,
      owner.accumExtSplats,
      visible.map((node, i) => [
        node.uuid,
        // introDots.time cambia solo la firma (rigenera), non il mapping (no sort)
        node.version - node.introDotsVersion,
        node.mappingVersion,
        counts[i],
      ]),
      meshes.map((mesh) => [
        this.identity(mesh.source),
        this.identity(mesh.primary),
        this.identity(mesh.secondary),
        ...mesh.sh.map((array) => this.identity(array)),
        this.identity(mesh.indices),
        mesh.indicesVersion,
        mesh.base,
        mesh.count,
        mesh.revision,
        mesh.sourceCount,
        mesh.numSh,
        mesh.rotation,
        mesh.translationScale,
        mesh.recolor,
        mesh.encoding,
        mesh.shMaxOpacity,
      ]),
    ]);
    const signature =
      view + mapping + JSON.stringify(meshes.map((mesh) => mesh.introDots));
    // A stale ordering still gets its final sort once the view settles.
    if (signature === this.currentSignature && !this.orderingStale) {
      next.dispose();
      return visible;
    }
    const sequence = ++this.sequence;
    next.version = sequence;
    next.mappingVersion = sequence;
    owner.sorting = true;
    this.backend.beginRenderSlotUpdate("native-generation");
    const generation = this.backend.getWorkingSlotGeneration();
    try {
      let activeSplats: number;
      let indirect = false;
      this.backend.generateNative({
        meshes,
        numSplats: next.numSplats,
        outputExt: next.extSplats,
        viewOrigin: next.viewOrigin.toArray(),
        viewDirection: next.viewDirection.toArray(),
        sortRadial,
        sortCullCos,
      });
      const rows = Math.max(1, Math.ceil(next.numSplats / 16384));
      const now = performance.now();
      // Like WebGL, which displays a regenerated accumulator with the last
      // ordering while its worker sort runs: one radix sort at a time (and no
      // more often than minSortIntervalMs) for a view-only change. Validity
      // is view-independent without the sort cone, or while the cone still
      // covers the whole view. Like PlayCanvas (gsplat-manager.js), a view
      // within SORT_EPSILON of the last sort keeps its ordering for good.
      const angle = Math.acos(
        Math.min(1, Math.max(-1, next.viewDirection.dot(this.lastSortDir))),
      );
      const withinEpsilon = sortRadial
        ? next.viewOrigin.distanceTo(this.lastSortOrigin) <= SORT_EPSILON
        : angle <= SORT_EPSILON;
      const reusable =
        sortMode === "radix" &&
        next.numSplats > 0 &&
        mapping === this.sortedMapping &&
        signature !== this.currentSignature &&
        (sortCullCos === -2 ||
          coneCoversView(camera, sortCullCos, angle, owner.sortCullFov));
      const reuseUntilSorted =
        reusable &&
        !withinEpsilon &&
        (this.backend.sortInFlight ||
          now < this.lastSortTime + owner.minSortIntervalMs);
      const reuse =
        reusable &&
        (withinEpsilon || reuseUntilSorted) &&
        this.backend.reuseActiveOrdering(rows);
      if (reuse) {
        indirect = true;
        activeSplats = meshes.reduce((sum, mesh) => sum + mesh.count, 0);
      } else if (
        sortMode === "radix" &&
        next.numSplats > 0 &&
        (await this.backend.gpuDepthSortAndPack({
          numSplats: next.numSplats,
          viewOrigin: next.viewOrigin,
          viewDir: next.viewDirection,
          sortRadial,
          enableExtSplats: next.extSplats,
          orderingRows: rows,
          orderingTarget: "working",
        }))
      ) {
        indirect = true;
        activeSplats = meshes.reduce((sum, mesh) => sum + mesh.count, 0);
        this.sortedMapping = mapping;
        this.lastSortTime = now;
        this.lastSortOrigin.copy(next.viewOrigin);
        this.lastSortDir.copy(next.viewDirection);
      } else {
        this.sortedMapping = "";
        const readback = await this.backend.readNativeDepths(next.numSplats);
        if (this.disposed || sequence !== this.sequence)
          throw new Error(
            "SparkRenderer is disposed or native generation was superseded",
          );
        if (this.ordering.length < rows * 16384)
          this.ordering = new Uint32Array(rows * 16384);
        this.worker ??= new SplatWorker();
        const result = await this.worker.call("sortSplats32", {
          numSplats: next.numSplats,
          readback,
          ordering: this.ordering,
        });
        this.ordering = result.ordering;
        if (this.disposed || sequence !== this.sequence)
          throw new Error(
            "SparkRenderer is disposed or native generation was superseded",
          );
        activeSplats = result.activeSplats;
        this.backend.uploadOrdering(result.ordering, 4096, rows, "working");
        this.backend.useIndirectDraw = false;
      }
      if (
        this.disposed ||
        sequence !== this.sequence ||
        generation !== this.backend.getWorkingSlotGeneration()
      )
        throw new Error("Spark: stale native generation");
      this.backend.commitRenderSlot();
      owner.activeSplats = activeSplats;
      this.stats = {
        generation: sequence,
        selectedSplats: meshes.reduce((sum, mesh) => sum + mesh.count, 0),
        drawnSplats: indirect ? null : activeSplats,
        ordering: indirect ? "radix" : "readback",
      };
      const old = owner.display;
      owner.current = next;
      owner.display = next;
      old.dispose();
      owner.sortedCenter.copy(next.viewOrigin);
      owner.sortedDir.copy(next.viewDirection);
      owner.setDirty();
      this.currentSignature = signature;
      // The final sort once the view settles (not needed within epsilon).
      this.orderingStale = reuse && reuseUntilSorted;
      if (this.orderingStale) this.scheduleFinalSort();
      return visible;
    } catch (error) {
      this.sortedMapping = "";
      this.backend.cancelRenderSlotUpdate(generation);
      next.dispose();
      throw error;
    } finally {
      owner.sorting = false;
    }
  }

  /** On-demand apps re-render (and so sort) once the next sort may run. */
  private scheduleFinalSort() {
    clearTimeout(this.staleTimeout);
    const poll = () => {
      this.staleTimeout = undefined;
      if (this.disposed || !this.orderingStale) return;
      const wait =
        this.lastSortTime + this.owner.minSortIntervalMs - performance.now();
      if (this.backend.sortInFlight || wait > 0) {
        this.staleTimeout = setTimeout(poll, Math.max(1, wait));
        return;
      }
      this.owner.setDirty();
    };
    this.staleTimeout = setTimeout(poll, 1);
  }

  /** Paged meshes read SplatPager's page pool through their LoD selection. */
  private pagedGeneration(
    node: SplatMesh,
    source: PagedSplats,
    base: number,
    count: number,
  ): NativeMeshGeneration {
    const pager = source.pager;
    const indices = source.dynoIndices.value.image.data as Uint32Array;
    if (!pager || indices.length < count)
      throw new Error("Spark: incomplete native paged selection");
    const extended = pager.extSplats;
    const enc = extended ? undefined : source.splatEncoding;
    const transform = node.context.transform;
    return {
      source,
      revision: pager.uploadVersion,
      primary: pager.packedTexture.value.image.data as Uint32Array,
      sh: [],
      sourceCount: pager.maxSplats,
      sourceExt: extended,
      base,
      count,
      indices,
      indicesVersion: source.indicesVersion,
      rotation: transform.rotate.value.toArray(),
      translationScale: [
        ...transform.translate.value.toArray(),
        transform.scale.value,
      ],
      localViewOrigin: node.context.viewToObject.translate.value.toArray(),
      encoding: [
        enc?.rgbMin ?? 0,
        enc?.rgbMax ?? 1,
        enc?.lnScaleMin ?? -12,
        enc?.lnScaleMax ?? 9,
      ],
      shMaxOpacity: [
        enc?.sh1Max ?? 1,
        enc?.sh2Max ?? 1,
        enc?.sh3Max ?? 1,
        enc?.lodOpacity ? 2 : 1,
      ],
      // Same clamp as PagedSplats.fetchSplat(); curSh guards empty textures.
      numSh: Math.min(
        node.maxSh,
        source.numSh,
        source.maxSh,
        pager.maxSh,
        pager.curSh,
      ),
      recolor: node.context.recolor.value.toArray(),
      introDots: node.introDots.active ? node.introDots.time : -1,
      pager,
      edits: this.meshEdits(node),
    };
  }

  /** Snapshot of the compiled SplatEdits that SplatMesh.update() just encoded. */
  private meshEdits(node: SplatMesh): NativeMeshGeneration["edits"] {
    const edits = node.rgbaDisplaceEdits;
    if (!edits?.numEdits) return undefined;
    const editWords = edits.numEdits * 4;
    const data = new Uint32Array(editWords + edits.numSdfs * 32);
    data.set(edits.editData.subarray(0, editWords));
    data.set(edits.sdfData.subarray(0, edits.numSdfs * 32), editWords);
    return { data, numEdits: edits.numEdits, numSdfs: edits.numSdfs };
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.sequence++;
    clearTimeout(this.staleTimeout);
    this.worker?.dispose();
    this.worker = null;
    this.composite?.dispose();
    this.composite = null;
    this.targetComposite?.dispose();
    this.targetComposite = null;
    this.backend.dispose();
  }
}
