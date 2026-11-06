import * as THREE from "three";
import { ExtSplats } from "./ExtSplats";
import { type LodMetaData, lodMetaFovScale, selectLodMeta } from "./LodMeta";
import { OrderingBufferPool } from "./OrderingBufferPool";
import { OrderingTrace } from "./OrderingTrace";
import { PackedSplats } from "./PackedSplats";
import { PagedSplats } from "./PagedSplats";
import { Readback } from "./Readback";
import {
  type RendererAdapter,
  type SparkHostRenderer,
  createRendererAdapter,
  isWebGLRenderer,
} from "./RendererAdapter";
import { getSparkRendererCapabilities } from "./RendererCapabilities";
import { SparkNativeRenderer } from "./SparkNativeRenderer";
import type { SparkRenderStats } from "./SparkRenderStats";
import { SplatAccumulator } from "./SplatAccumulator";
import { isSplatEdit } from "./SplatEdit";
import type { SplatGenerator } from "./SplatGenerator";
import { SplatGeometry } from "./SplatGeometry";
import { SplatMesh } from "./SplatMesh";
import { SplatPager } from "./SplatPager";
import { SplatWorker } from "./SplatWorker";
import { SPLAT_TEX_HEIGHT, SPLAT_TEX_WIDTH } from "./defines";
import { SPARK_ENABLE_HOOKS, sparkHook } from "./hooks";
import { getShaders } from "./shaders";
import {
  type ParallelSortBuffers,
  SORT_PARALLEL_MIN_SPLATS,
  ensureParallelSortBuffers,
  parallelSort16,
  parallelSort32,
} from "./sortParallel";
import {
  isAndroid,
  isIos,
  isMobile,
  isOculus,
  isVisionPro,
  uploadU32DataTextureRows,
} from "./utils";
import * as wasm from "./wasm";
import type { rpcHandlers } from "./worker";

// Tolerance for stochastic pose comparison: absorbs SparkControls' decaying
// inertia tail (never exactly 0) without mistaking it for real movement.
const STOCHASTIC_POSE_EPSILON = 1e-5;

export interface SparkRendererOptions<
  R extends SparkHostRenderer = THREE.WebGLRenderer,
> {
  /**
   * Pass in your THREE.WebGLRenderer instance so Spark can perform work
   * outside the usual render loop. Should be created with antialias: false
   * (default setting) as WebGL anti-aliasing doesn't improve Gaussian Splatting
   * rendering and significantly reduces performance.
   */
  renderer: R;
  /**
   * Native sort implementation. "radix" (default) sorts on the GPU with no
   * readback; "readback" uses GPU-generated depths and WASM sorting on the CPU,
   * which stalls every frame while the camera moves.
   */
  webgpuSort?: "readback" | "radix";
  /**
   * Callback function to be called when SparkRenderer needs to re-render,
   * for example when splat sort order or LoD updates complete. May fire
   * several times per frame; schedule a single render rather than rendering
   * inside the callback.
   */
  onDirty?: () => void;
  /**
   * Whether to use premultiplied alpha when accumulating splat RGB
   * @default true
   */
  premultipliedAlpha?: boolean;
  /**
   * Pass in a THREE.Timer to to synchronize time-based effects across different
   * systems.
   */
  timer?: THREE.Timer;
  /**
   * Controls whether to check and automatically update Gsplat collection
   * each frame render.
   * @default true
   */
  autoUpdate?: boolean;
  /**
   * Controls whether to update the Gsplats before or after rendering. For WebXR
   * this is set to false in order to complete rendering as soon as possible.
   * @default true (if not WebXR)
   */
  preUpdate?: boolean;
  /**
   * Maximum standard deviations from the center to render Gaussians. Values
   * Math.sqrt(4)..Math.sqrt(9) produce acceptable results and can be tweaked for
   * performance.
   * @default Math.sqrt(8)
   */
  maxStdDev?: number;
  /*
   **
   * Minimum pixel radius for splat rendering.
   * @default 0.0
   */
  minPixelRadius?: number;
  /**
   * Maximum pixel radius for splat rendering.
   * @default 512.0
   */
  maxPixelRadius?: number;
  /**
   * Whether to use extended Gsplat encoding for intermediary accumulator splats.
   * @default false
   */
  accumExtSplats?: boolean;
  /**
   * Whether to use covariance Gsplat encoding for intermediary splats.
   * @default false
   */
  covSplats?: boolean;
  /**
   * Minimum alpha value for splat rendering.
   * @default 0.5 * (1.0 / 255.0)
   */
  minAlpha?: number;
  /**
   * Enable 2D Gaussian splatting rendering ability. When this mode is enabled,
   * any scale x/y/z component that is exactly 0 (minimum quantized value) results
   * in the other two non-0 axis being interpreted as an oriented 2D Gaussian Splat,
   * rather instead of the usual projected 3DGS Z-slice. When reading PLY files,
   * scale values less than e^-30 will be interpreted as 0.
   * @default false
   */
  enable2DGS?: boolean;
  /**
   * Enable alternative ray-splat max response evaluation, used by 3DGUT (unscented transform),
   * 3DGRT, and HTGS.
   * @default false
   */
  // enableRayEval?: boolean;
  /**
   * Scalar value to add to 2D splat covariance diagonal, effectively blurring +
   * enlarging splats. In scenes trained without the Gsplat anti-aliasing tweak
   * this value was typically 0.3, but with anti-aliasing it is 0.0
   * @default 0.0
   */
  preBlurAmount?: number;
  /**
   * Scalar value to add to 2D splat covarianve diagonal, with opacity adjustment
   * to correctly account for "blurring" when anti-aliasing. Typically 0.3
   * (equivalent to approx 0.5 pixel radius) in scenes trained with anti-aliasing.
   */
  blurAmount?: number;
  /**
   * Depth-of-field distance to focal plane
   */
  focalDistance?: number;
  /**
   * Full-width angle of aperture opening (in radians), 0.0 to disable
   * @default 0.0
   */
  apertureAngle?: number;
  /**
   * Modulate Gaussian kernel falloff. 0 means "no falloff, flat shading",
   * while 1 is the normal Gaussian kernel.
   * @default 1.0
   */
  falloff?: number;
  /**
   * X/Y clipping boundary factor for Gsplat centers against view frustum.
   * 1.0 clips any centers that are exactly out of bounds, while 1.4 clips
   * centers that are 40% beyond the bounds.
   * @default 1.4
   */
  clipXY?: number;
  /**
   * Parameter to adjust projected splat scale calculation to match other renderers,
   * similar to the same parameter in the MKellogg 3DGS renderer. Higher values will
   * tend to sharpen the splats. A value 2.0 can be used to match the behavior of
   * the PlayCanvas renderer.
   * @default 1.0
   */
  focalAdjustment?: number;
  /**
   * Whether to sort splats radially (geometric distance) from the viewpoint (true)
   * or by Z-depth (false). Most scenes are trained with the Z-depth `sort `metric
   * and will render more accurately at certain viewpoints. However, radial sorting
   * is more stable under viewpoint rotations.
   * @default true
   */
  sortRadial?: boolean;
  /**
   * Minimum interval between sort calls in milliseconds.
   * @default 0
   */
  minSortIntervalMs?: number;
  /**
   * Number of workers for the depth sort. Above 1 the sort is split across
   * that many workers via SharedArrayBuffer, used only when the page is
   * crossOriginIsolated and there are at least 1M splats; otherwise a single
   * worker sorts.
   * @default max(1, min(4, hardwareConcurrency - 1))
   */
  sortThreads?: number;
  /**
   * Sort only splats whose center lies inside a cone around the view
   * direction, with this full angle in degrees; the others are not drawn.
   * Use a value larger than the camera's diagonal FOV (splats are culled by
   * center, and the sort runs behind the camera motion). Ignored for
   * orthographic cameras. 0 disables it.
   * @default 0
   */
  sortCullFov?: number;
  /**
   * WebGPU native path without tone mapping: blend splats in sRGB space like
   * the WebGL path (and other 3DGS viewers) instead of in linear space.
   * Three.js objects keep their linear blending. Ignored with tone mapping.
   * @default true
   */
  srgbBlend?: boolean;
  /**
   * Sort by the depth rounded to 16-bit float (one counting pass instead of
   * two radix passes). Same splats are drawn, ordering precision is ~1/2048
   * of the distance.
   * @default false
   */
  sort16?: boolean;
  /**
   * Sort-free stochastic rendering: each splat fragment is kept with
   * probability alpha and written opaque with depth, drawn front-to-back
   * from the last sorted order. Noisy but order-independent, so a stale sort
   * while moving shows no popping. "movement" uses it only while the camera
   * moves and returns to the sorted render stochasticSettleMs after it stops.
   * Applied only to renders to the canvas (or the XR framebuffer); renders
   * to other render targets always use the sorted path.
   * @default "off"
   */
  stochastic?: "off" | "on" | "movement";
  /**
   * With stochastic "movement": milliseconds without camera changes before
   * switching back to the sorted render.
   * @default 100
   */
  stochasticSettleMs?: number;
  /*
   * Flag to control whether LoD is enabled. @default true
   */
  enableLod?: boolean;
  /**
   * Whether to drive LOD updates (compute lodInstances, update pager, etc.).
   * Set to false to use LOD instances from another renderer without driving updates.
   * Only has effect if enableLod is true.
   * @default true (if enableLod is true)
   */
  enableDriveLod?: boolean;
  /**
   * Whether to enable page fetching for LoD.
   * @default true
   */
  enableLodFetching?: boolean;
  /**
   * Set the target # splats for LoD. If this isn't set then default base LoD splat
   * counts will apply: 500K-750K for WebXR, 1-1.5M for mobile, and 2.5M for desktop.
   * @default 500K-2500K depending on platform
   */
  lodSplatCount?: number;
  /**
   * Scale factor for target # splats for LoD. 2.0 means 2x the base LoD splat count.
   * This is the easiest LoD parameter to adjust and will scale detail appropriately
   * for the platform.
   * @default 1.0
   */
  lodSplatScale?: number;
  /**
   * Determines the minimum screen pixel size of LoD splats. The default 1.0 means
   * the splat LoD tree will pick splats that are no smaller than 1 pixel in size.
   * Setting this to a higher value as high as 5.0 will often be indistinguishable
   * but will avoid wasting rendering capacity on tiny splats.
   * @default 1.0
   */
  lodRenderScale?: number;
  /**
   * Inflate LoD splats to ensure opacity stays <= 1.0, producing a softer appearance.
   * @default false
   */
  lodInflate?: boolean;
  lodTraverseMode?: "dynamic" | "standard";
  /**
   * For lod-meta.json (streamed SOG) meshes: limita il dettaglio massimo al
   * livello deciso dalle distanze (c=0), il budget splat può solo abbassarlo.
   * @default true
   */
  lodMetaLimit?: boolean;
  /**
   * Whether to use extended Gsplat encoding for paged splats, useful for eliminating
   * quantization artifacts from splat scenes with large internal position coordinates.
   * @default false
   */
  pagedExtSplats?: boolean;
  /**
   * Allocation size of paged splats. This must be a multiple of the page size (65536).
   * @default 16777216 (256 * 65536) for desktop, 6291456 for iOS, 8,388,608 for other mobile
   */
  maxPagedSplats?: number;
  /**
   * Number of parallel chunk fetchers for LoD. These are run within a shared pool
   * of 4 background WebWorker threads, so setting it above 4 will not have any
   * effect. Setting it 3 leaves one spare worker for other loading/decoding tasks.
   * @default 3
   */
  numLodFetchers?: number;
  /**
   * How long (ms) a LoD SplatMesh can go unrendered (hidden or removed from the
   * scene) before its LoD state is released: the worker-side tree is dropped and,
   * for paged meshes, its resident pages are freed for other meshes. Rendering it
   * again rebuilds the tree and refetches pages. Set to Infinity to never release.
   * @default 3000
   */
  lodCleanupTimeoutMs?: number;
  /**
   * Full-width angle in degrees of fixed foveation cone along the view direction
   * with no foveation applied (full resolution, foveate=1.0). Set to 0 to disable.
   * @default 90.0
   */
  coneFov0?: number;
  /**
   * Full-width angle in degrees of fixed foveation cone along the view direction
   * with reduced resolution specified by `coneFoveate`. Foveation will be applied
   * smoothly from 1.0 down to `coneFoveate` as you move outward from
   * `coneFov0` to `coneFov`. Set to 0 to disable.
   * @default 120.0
   */
  coneFov?: number;
  /**
   * Foveation scale to apply to LoD splats at the edge of coneFov. Foveation will
   * be applied smoothly from `coneFoveate` down to `behindFoveate` as you move
   * outward from `coneFov` to 180 degrees (behind the viewer).
   * @default 0.4
   */
  coneFoveate?: number;
  /**
   * Foveation scale to apply to LoD splats behind the viewer. Setting this to 0.1
   * for example will result in splats 10x larger than inside the viewing frustum.
   * @default 0.2
   */
  behindFoveate?: number;
  /**
   * How many LoD splats to generate for raycasting
   * @default 10000-25000 iff default canvas target is used
   */
  lodRaycast?: number;
  lodRaycastIntervalMs?: number;
  /**
   * Configures an offline render target for the SparkRenderer (as opposed to
   * rendering to the canvas). This is useful for rendering environment maps,
   * additional viewpoints, or video frame rendering.
   * @default undefined
   */
  target?: {
    /**
     * Width of the render target in pixels.
     */
    width: number;
    /**
     * Height of the render target in pixels.
     */
    height: number;
    /**
     * If you want to be able to render a scene that depends on this target's
     * output (for example, a recursive viewport), set this to true to enable
     * double buffering.
     * @default false
     */
    doubleBuffer?: boolean;
    /**
     * Super-sampling factor for the render target. Values 1-4 are supported.
     * Note that re-sampling back down to .width x .height is done on the CPU
     * with simple averaging only when calling readTarget().
     * @default 1
     */
    superXY?: number;
  } & THREE.RenderTargetOptions;
  /**
   * Extra uniform values to pass to the shader.
   * @default undefined = no extra uniforms
   */
  extraUniforms?: Record<string, unknown>;
  /**
   * Replace the default `splatVertex.glsl` splat shader with a custom one.
   * @default undefined = use the default `splatVertex.glsl` shader
   */
  vertexShader?: string;
  /**
   * Replace the default `splatFragment.glsl` splat shader with a custom one.
   * @default undefined = use the default `splatFragment.glsl` shader
   */
  fragmentShader?: string;
  /**
   * Set the splat shader material to be transparent which determines if the
   * splats are rendered during the first opaque THREE.js render pass or the
   * second transparent render pass.
   * @default undefined = true
   */
  transparent?: boolean;
  /**
   * Set the splat shader material to enable depth testing which determines if the
   * splats respect the Z depth buffer and blend with other opaque objects in the scene.
   * @default undefined = true
   */
  depthTest?: boolean;
  /**
   * Set the splat shader material to enable depth writing which determines if the
   * splats write to the Z depth buffer. Note that enabling this may produce
   * undesirable results because most of the Gsplat is transparent.
   * @default undefined = false
   */
  depthWrite?: boolean;
}

/**
 * Result of {@link SparkRenderer.pickAsync}. Depths/distances are linear
 * view-space depth (world units along the camera's forward axis), not
 * Euclidean distance from the camera.
 */
export interface SparkPickResult {
  /** Best point: always pointMedia (pointSolida is extra info only). */
  point: THREE.Vector3;
  /** World point reconstructed from the alpha-weighted average depth. */
  pointMedia: THREE.Vector3;
  /** World point reconstructed from the first "solid" splat's depth, if any
   *  (always null unless pickAsync() was called with solid: true). */
  pointSolida: THREE.Vector3 | null;
  distanceMedia: number;
  distanceSolida: number | null;
  /** Sum of alpha coverage (G channel) at the chosen pixel. */
  copertura: number;
  floatType: "float" | "half";
  /** Wall-clock time for the whole pickAsync() call, in ms. */
  ms: number;
}

export class SparkRenderer<
  R extends SparkHostRenderer = THREE.WebGLRenderer,
> extends THREE.Mesh {
  readonly renderer: R;
  readonly material: R extends THREE.WebGLRenderer
    ? THREE.ShaderMaterial
    : THREE.Material;
  private readonly adapter: RendererAdapter;
  private readonly native: SparkNativeRenderer | null;
  webgpuSort: "readback" | "radix";

  get isWebGPU() {
    return this.adapter.kind === "webgpu";
  }
  get webgpuBackend() {
    return this.native?.backend ?? null;
  }
  getCapabilities() {
    return getSparkRendererCapabilities(this.renderer);
  }

  /** Committed counts. GPU radix counts remain unavailable until explicitly read. */
  getRenderStats(): SparkRenderStats {
    return (
      this.native?.getRenderStats() ?? {
        generation: this.display.version,
        selectedSplats: this.display.mapping.reduce(
          (sum, mesh) => sum + mesh.count,
          0,
        ),
        drawnSplats: this.activeSplats,
        ordering: "readback",
      }
    );
  }

  /** Read counts for the generation committed when called; this may wait for the GPU. */
  readRenderStatsAsync(): Promise<SparkRenderStats> {
    if (this.isDisposed)
      return Promise.reject(new Error("SparkRenderer is disposed"));
    return (
      this.native?.readRenderStatsAsync() ??
      Promise.resolve(this.getRenderStats())
    );
  }

  private requireWebGLRenderer(): THREE.WebGLRenderer {
    if (this.adapter.kind !== "webgl")
      throw new Error("Spark: this operation requires THREE.WebGLRenderer");
    return this.adapter.renderer;
  }
  readonly uniforms: ReturnType<typeof SparkRenderer.makeUniforms>;

  autoUpdate: boolean;
  preUpdate: boolean;
  static sparkOverride?: SparkRenderer<SparkHostRenderer>;

  renderSize = new THREE.Vector2();
  maxStdDev: number;
  minPixelRadius: number;
  maxPixelRadius: number;
  accumExtSplats: boolean;
  covSplats: boolean;
  minAlpha: number;
  enable2DGS: boolean;
  // enableRayEval: boolean;
  preBlurAmount: number;
  blurAmount: number;
  focalDistance: number;
  apertureAngle: number;
  falloff: number;
  clipXY: number;
  focalAdjustment: number;

  sortRadial: boolean;
  minSortIntervalMs: number;
  sortThreads: number;
  sortCullFov: number;
  srgbBlend: boolean;
  sort16: boolean;
  // sortCullFov of the last generated accumulator, as cos(half angle)
  private sortCullCos = -2;

  stochastic: "off" | "on" | "movement";
  stochasticSettleMs: number;
  // camera matrixWorld + projectionMatrix of the last canvas frame
  // (float64: three's matrix elements are float64, a float32 cache would
  // round every value and always compare as "moved")
  private stochasticPose = new Float64Array(32);
  private stochasticMoveTime = Number.NEGATIVE_INFINITY;
  private stochasticTimeoutId: ReturnType<typeof setTimeout> | undefined =
    undefined;
  // Decided once per canvas frame, reused by XR secondary eyes
  /** @internal Also read by the native WebGPU renderer. */
  stochasticActive = false;
  // Material state overridden for the current draw, restored in onAfterRender
  private stochasticSaved: {
    blending: THREE.Blending;
    depthWrite: boolean;
    depthTest: boolean;
  } | null = null;

  readonly timer: THREE.Timer;
  private readonly ownsTimer: boolean;
  lastFrame = -1;
  updateTimeoutId: ReturnType<typeof setTimeout> | undefined = undefined;
  onDirty?: () => void;
  dirty: boolean;

  private readonly orderingBuffers = new OrderingBufferPool();
  orderingTexture: THREE.DataTexture | null = null;
  maxSplats = 0;
  /**
   * WebGL/reference draw count. With native radix this is an upper bound until
   * readRenderStatsAsync() completes; use getRenderStats() for explicit validity.
   */
  activeSplats = 0;

  display: SplatAccumulator;
  current: SplatAccumulator;
  accumulators: SplatAccumulator[] = [];

  private isDisposed = false;

  sorting = false;
  sortDirty = false;
  // Mapping version the latest update produced.
  private latestMappingVersion = -1;
  lastSortTime = 0;
  sortWorker: SplatWorker | null = null;
  sortTimeoutId: ReturnType<typeof setTimeout> | undefined = undefined;
  sortedCenter = new THREE.Vector3().setScalar(Number.NEGATIVE_INFINITY);
  sortedDir = new THREE.Vector3().setScalar(0);
  readback32 = new Uint32Array(0);
  // Parallel sort: dedicated workers, shared buffers, upload copy of ordering
  sortPool: SplatWorker[] = [];
  private parallelSortBuffers?: ParallelSortBuffers;

  // Meshes still loading; a render is requested when they finish.
  private readonly initWatched = new WeakSet<SplatMesh>();

  enableLod: boolean;
  enableDriveLod: boolean;
  enableLodFetching: boolean;
  lodSplatCount?: number;
  lodSplatScale: number;
  lodRenderScale: number;
  lodInflate: boolean;
  lodTraverseMode: "dynamic" | "standard";
  lodMetaLimit: boolean;
  pagedExtSplats: boolean;
  maxPagedSplats: number;
  numLodFetchers: number;
  lodCleanupTimeoutMs: number;
  behindFoveate: number;
  coneFov0: number;
  coneFov: number;
  coneFoveate: number;

  lodRaycast?: number;
  lodRaycastIntervalMs: number;
  lastLodRaycastTime = 0;

  lodWorker: SplatWorker | null = null;
  // Raycast-only copy of the LoD trees (raycastAsync), so picks never queue
  // behind traversals. Gets every tree create/update/dispose of lodWorker, with
  // copied buffers and in the same order, so both assign the same lodIds.
  pickWorker: SplatWorker | null = null;
  // raycastPickWorker() state: one request in flight, FIFO of waiting ones,
  // latest raycastAsync() generation per coalesce key
  private pickBusy = false;
  private pickQueue: { key?: string; go: (go: boolean) => void }[] = [];
  private pickCounter = 0;
  private pickLatest = new Map<string, number>();
  private disposed = false;
  lodMeshes: { mesh: SplatMesh; version: number }[] = [];
  lodDirty = false;
  // Fork: page-arrival traversals coalesced to at most one per 300 ms
  private lodPagesPending = false;
  private lastPagesLodTime = 0;
  private lodPagesTimer?: ReturnType<typeof setTimeout>;
  lodIds: Map<
    PackedSplats | ExtSplats | PagedSplats,
    { lodId: number; lastTouched: number; rootPage?: number }
  > = new Map();
  lodIdToSplats: Map<number, PackedSplats | ExtSplats | PagedSplats> =
    new Map();
  lodInitQueue: (PackedSplats | ExtSplats | PagedSplats)[] = [];
  private lodInitFailed = new WeakSet<PackedSplats | ExtSplats | PagedSplats>();
  lastLod?: {
    pos: THREE.Vector3;
    quat: THREE.Quaternion;
    pixelScaleLimit: number;
    maxSplats: number;
    timestamp: number;
  };
  currentLod?: {
    pos: THREE.Vector3;
    quat: THREE.Quaternion;
    pixelScaleLimit: number;
    maxSplats: number;
    timestamp: number;
  };
  lodPosOverride?: THREE.Vector3;
  lodQuatOverride?: THREE.Quaternion;
  lodInstances: Map<
    SplatMesh,
    {
      lodId: number;
      numSplats: number;
      indices: Uint32Array;
      texture: THREE.DataTexture;
    }
  > = new Map();
  lodUpdates: {
    lodId: number;
    pageBase: number;
    chunkBase: number;
    count: number;
    lodTreeData?: Uint32Array;
  }[] = [];
  lastTraverseTime = 0;
  lastPixelLimit?: number;
  // lod-meta.json: riselezione con cadenza e soglie PlayCanvas (gsplat-world.js
  // testCameraMovedForLod): ogni 10 frame, solo se camera >0,5 m, >90° o FOV >2%
  private lodMetaDirty = true;
  private lodMetaFrames = 0;
  private lodMetaFovScale = 1;
  private lodMetaLast?: {
    pos: THREE.Vector3;
    fwd: THREE.Vector3;
    fovScale: number;
  };
  private lodMetaPriority: { splats: PagedSplats; chunk: number }[] = [];
  private lodMetaPrefetch: { splats: PagedSplats; chunk: number }[] = [];

  pager?: SplatPager;
  pagerId = 0;
  // prefetchCameras: THREE.Camera[] = [];
  // prefetchLodScale = 1.0;
  // prefetchMeshesCache: SplatMesh[] = [];
  // prefetchMeshesCacheScene?: THREE.Scene;

  target?: THREE.WebGLRenderTarget;
  backTarget?: THREE.WebGLRenderTarget;
  superPixels?: Uint8Array;
  targetPixels?: Uint8Array;
  superXY = 1;

  // GPU pick (pickAsync) state: lazily created materials/targets, reused and
  // only recreated when size or float-vs-half precision changes. Two
  // passes are needed (see pickAsync()): a weighted-average pass
  // (premultiplied "over" blend) and a first-solid-hit pass (MIN blend).
  private pickMaterialMedia?: THREE.ShaderMaterial;
  private pickMaterialSolid?: THREE.ShaderMaterial;
  private pickTargetMedia?: THREE.WebGLRenderTarget;
  private pickTargetSolid?: THREE.WebGLRenderTarget;
  private pickTargetW = 0;
  private pickTargetH = 0;
  private pickTargetIsFloat = false;
  // Set for the duration of the synchronous pickAsync() render call so
  // onBeforeRender() skips the sort/LoD-update trigger (see onBeforeRender);
  // everything else in onBeforeRender (per-draw uniforms) still runs so the
  // pick pass sees the viewport/camera it's actually given.
  private pickPass = false;

  flushAfterGenerate = false;
  flushAfterRead = false;
  readPause = 1;
  sortPause = 0;
  sortDelay = 0;

  constructor(options: SparkRendererOptions<R>) {
    if (!options) {
      throw new Error("SparkRenderer options are required");
    }
    if (!options.renderer) {
      throw new Error("renderer is required in SparkRenderer options");
    }

    const adapter = createRendererAdapter(options.renderer);
    if (
      adapter.kind === "webgpu" &&
      (options.target ||
        options.vertexShader ||
        options.fragmentShader ||
        options.extraUniforms ||
        options.depthWrite ||
        options.depthTest === false ||
        options.transparent === false)
    ) {
      throw new Error(
        "Spark: custom targets, shaders and material overrides require THREE.WebGLRenderer.",
      );
    }
    const uniforms = SparkRenderer.makeUniforms();
    Object.assign(uniforms, options.extraUniforms ?? {});

    const shaders = getShaders();
    const premultipliedAlpha = options.premultipliedAlpha ?? true;
    const geometry = new SplatGeometry();
    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: options.vertexShader ?? shaders.splatVertex,
      fragmentShader: options.fragmentShader ?? shaders.splatFragment,
      uniforms,
      premultipliedAlpha,
      transparent: options.transparent ?? true,
      depthTest: options.depthTest ?? true,
      depthWrite: options.depthWrite ?? false,
      side: THREE.DoubleSide,
      allowOverride: false,
    });

    super(geometry, material);
    this.adapter = adapter;
    this.webgpuSort = options.webgpuSort ?? "radix";
    if (adapter.kind === "webgpu") {
      geometry.dispose();
      material.dispose();
      this.geometry = new THREE.BufferGeometry();
      // Hide only the placeholder draw, preserving visibility of child splats.
      this.material = new THREE.MeshBasicMaterial({
        visible: false,
        premultipliedAlpha,
      }) as THREE.Material as SparkRenderer<R>["material"];
    } else {
      this.material = material as SparkRenderer<R>["material"];
    }
    this.native =
      adapter.kind === "webgpu"
        ? new SparkNativeRenderer(this, adapter.renderer, adapter.device)
        : null;
    this.uniforms = uniforms;
    // Disable frustum culling because we want to always draw them all
    // and cull Gsplats individually in the shader
    this.frustumCulled = false;

    // By default SparkRenderer will only render for layer 0
    // this.layers.enableAll();

    // sparkRendererInstance = this;
    this.renderer = options.renderer;
    this.onDirty = options.onDirty;
    this.dirty = true;
    this.autoUpdate = options.autoUpdate ?? true;
    this.preUpdate = options.preUpdate ?? true;

    this.maxStdDev = options.maxStdDev ?? Math.sqrt(8.0);
    this.minPixelRadius = options.minPixelRadius ?? 0.0; //1.6;
    this.maxPixelRadius = options.maxPixelRadius ?? 512.0;
    this.accumExtSplats = options.accumExtSplats ?? false;
    this.covSplats = options.covSplats ?? false;
    this.minAlpha = options.minAlpha ?? 0.5 * (1.0 / 255.0);
    this.enable2DGS = options.enable2DGS ?? false;
    // this.enableRayEval = options.enableRayEval ?? false;
    this.preBlurAmount = options.preBlurAmount ?? 0.0;
    this.blurAmount = options.blurAmount ?? 0.3;
    this.focalDistance = options.focalDistance ?? 0.0;
    this.apertureAngle = options.apertureAngle ?? 0.0;
    this.falloff = options.falloff ?? 1.0;
    this.clipXY = options.clipXY ?? 1.4;
    this.focalAdjustment = options.focalAdjustment ?? 1.0;

    this.sortRadial = options.sortRadial ?? true;
    this.minSortIntervalMs = options.minSortIntervalMs ?? 0;
    this.sortThreads =
      options.sortThreads ??
      Math.max(
        1,
        Math.min(4, (globalThis.navigator?.hardwareConcurrency || 2) - 1),
      );
    this.sortCullFov = options.sortCullFov ?? 0;
    this.srgbBlend = options.srgbBlend ?? true;
    this.sort16 = options.sort16 ?? false;
    this.stochastic = options.stochastic ?? "off";
    this.stochasticSettleMs = options.stochasticSettleMs ?? 100;

    this.enableLod = options.enableLod ?? true;
    // enableDriveLod defaults to true if enableLod is true, false otherwise
    this.enableDriveLod = options.enableDriveLod ?? this.enableLod;
    this.enableLodFetching = options.enableLodFetching ?? true;
    this.lodSplatCount = options.lodSplatCount;
    this.lodSplatScale = options.lodSplatScale ?? 1.0;
    this.lodRenderScale = options.lodRenderScale ?? 1.0;
    this.lodInflate = options.lodInflate ?? false;
    this.lodTraverseMode = options.lodTraverseMode ?? "standard";
    this.lodMetaLimit = options.lodMetaLimit ?? true;
    this.pagedExtSplats = options.pagedExtSplats ?? false;
    const defaultPages = isMobile() ? (isIos() ? 96 : 128) : 256;
    this.maxPagedSplats = options.maxPagedSplats ?? defaultPages * 65536;
    this.numLodFetchers = options.numLodFetchers ?? 3;
    this.lodCleanupTimeoutMs = options.lodCleanupTimeoutMs ?? 3000;
    this.behindFoveate = options.behindFoveate ?? 0.2;
    this.coneFov0 = options.coneFov0 ?? 90.0;
    this.coneFov = options.coneFov ?? 120.0;
    this.coneFoveate = options.coneFoveate ?? 0.4;

    this.lodRaycast =
      options.lodRaycast === undefined
        ? isMobile()
          ? 10000
          : 25000
        : options.lodRaycast;
    this.lodRaycastIntervalMs = options.lodRaycastIntervalMs ?? 500;

    this.timer = options.timer ?? new THREE.Timer();
    this.ownsTimer = !!options.timer;

    const accumulatorOptions = {
      extSplats: this.accumExtSplats,
      covSplats: this.covSplats,
    };
    this.display = new SplatAccumulator(accumulatorOptions);
    this.current = this.display;
    this.accumulators.push(new SplatAccumulator(accumulatorOptions));
    this.accumulators.push(new SplatAccumulator(accumulatorOptions));

    // Check if the provoking vertex convention should be changed
    const provokingVertexExt =
      adapter.kind === "webgl"
        ? adapter.renderer.getContext().getExtension("WEBGL_provoking_vertex")
        : null;
    if (provokingVertexExt) {
      provokingVertexExt.provokingVertexWEBGL(
        provokingVertexExt.FIRST_VERTEX_CONVENTION_WEBGL,
      );
    }

    if (options.target) {
      const {
        width,
        height,
        doubleBuffer,
        superXY: origSuperXY,
        ...origTargetOptions
      } = options.target;
      const superXY = Math.max(1, Math.min(4, origSuperXY ?? 1));
      if (width * superXY > 8192 || height * superXY > 8192) {
        throw new Error("Target size too large");
      }
      this.superXY = superXY;

      const superWidth = width * superXY;
      const superHeight = height * superXY;
      const targetOptions: THREE.RenderTargetOptions = {
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        colorSpace: THREE.SRGBColorSpace,
        ...origTargetOptions,
      };

      this.target = new THREE.WebGLRenderTarget(
        superWidth,
        superHeight,
        targetOptions,
      );
      if (doubleBuffer) {
        this.backTarget = new THREE.WebGLRenderTarget(
          superWidth,
          superHeight,
          targetOptions,
        );
      }
    }
  }

  static makeUniforms() {
    const uniforms = {
      // // number of active splats to render
      // numSplats: { value: 0 },
      // Size of render viewport in pixels
      renderSize: { value: new THREE.Vector2() },
      // Near and far plane distances
      near: { value: 0.1 },
      far: { value: 1000.0 },
      // SplatAccumulator to view transformation quaternion
      renderToViewQuat: { value: new THREE.Quaternion() },
      // SplatAccumulator to view transformation translation
      renderToViewPos: { value: new THREE.Vector3() },
      renderToViewBasis: { value: new THREE.Matrix3() },
      renderToViewOffset: { value: new THREE.Vector3() },
      // Maximum distance (in stddevs) from Gsplat center to render
      maxStdDev: { value: 1.0 },
      // Minimum pixel radius for splat rendering
      minPixelRadius: { value: 0.0 },
      // Maximum pixel radius for splat rendering
      maxPixelRadius: { value: 512.0 },
      // Minimum alpha value for splat rendering
      minAlpha: { value: 0.5 * (1.0 / 255.0) },
      // Enable interpreting 0-thickness Gsplats as 2DGS
      enable2DGS: { value: false },
      // Enable ray-splat max response evaluation
      // enableRayEval: { value: false },
      // Inflate LoD splats so that opacity <= 1.0
      lodInflate: { value: false },
      // Add to projected 2D splat covariance diagonal (thickens and brightens)
      preBlurAmount: { value: 0.0 },
      // Add to 2D splat covariance diagonal and adjust opacity (anti-aliasing)
      blurAmount: { value: 0.3 },
      // Depth-of-field distance to focal plane
      focalDistance: { value: 0.0 },
      // Full-width angle of aperture opening (in radians)
      apertureAngle: { value: 0.0 },
      // Modulate Gaussian kernal falloff. 0 means "no falloff, flat shading",
      // 1 is normal e^-x^2 falloff.
      falloff: { value: 1.0 },
      // Clip Gsplats that are clipXY times beyond the +-1 frustum bounds
      clipXY: { value: 1.4 },
      // Debug renderSize scale factor
      focalAdjustment: { value: 1.0 },
      // Whether to encode Gsplat with linear RGB (for environment mapping)
      encodeLinear: { value: false },
      // Back-to-front sort ordering of splat indices
      ordering: { type: "t", value: SparkRenderer.emptyOrdering },
      enableExtSplats: { value: false },
      enableCovSplats: { value: false },
      // Gsplat collection to render
      extSplats: { type: "t", value: SplatAccumulator.emptyTexture },
      extSplats2: { type: "t", value: SplatAccumulator.emptyTexture },
      // Time in seconds for time-based effects
      time: { value: 0 },
      // Delta time in seconds since last frame
      deltaTime: { value: 0 },
      // Debug flag that alternates each frame
      debugFlag: { value: false },
      // Sort-free stochastic rendering (front-to-back, alpha-tested)
      stochastic: { value: false },
      // Number of drawn instances, to reverse the ordering when stochastic
      instanceCount: { value: 0 },
    };
    return uniforms;
  }

  dispose() {
    if (this.isDisposed) return;
    this.isDisposed = true;
    this.orderingTrace.dispose();
    this.native?.dispose();

    // @ts-ignore Object3D has a dispose method in Three.js >= r186
    super.dispose?.();

    // No worker is created again: pending picks abort, later ones reject
    this.disposed = true;
    const queue = this.pickQueue;
    this.pickQueue = [];
    for (const { go } of queue) go(false);
    this.pickLatest.clear();
    this.lodIds.clear();
    this.lodIdToSplats.clear();

    if (this.updateTimeoutId !== undefined) {
      clearTimeout(this.updateTimeoutId);
      this.updateTimeoutId = undefined;
    }
    if (this.sortTimeoutId !== undefined) {
      clearTimeout(this.sortTimeoutId);
      this.sortTimeoutId = undefined;
    }
    if (this.stochasticTimeoutId !== undefined) {
      clearTimeout(this.stochasticTimeoutId);
      this.stochasticTimeoutId = undefined;
    }

    this.geometry.dispose();
    this.material.dispose();

    if (this.target) {
      this.target.dispose();
      this.target = undefined;
    }
    if (this.backTarget) {
      this.backTarget.dispose();
      this.backTarget = undefined;
    }
    if (this.pickTargetMedia) {
      this.pickTargetMedia.dispose();
      this.pickTargetMedia = undefined;
    }
    if (this.pickTargetSolid) {
      this.pickTargetSolid.dispose();
      this.pickTargetSolid = undefined;
    }
    this.pickMaterialMedia?.dispose();
    this.pickMaterialMedia = undefined;
    this.pickMaterialSolid?.dispose();
    this.pickMaterialSolid = undefined;
    if (this.orderingTexture) {
      this.orderingTexture.dispose();
      this.orderingTexture = null;
    }
    this.orderingBuffers.clear();

    const accumulators = new Set<SplatAccumulator>();
    accumulators.add(this.display);
    accumulators.add(this.current);
    for (const accumulator of this.accumulators) {
      accumulators.add(accumulator);
    }
    for (const accumulator of accumulators) {
      accumulator.dispose();
    }

    for (const instance of this.lodInstances.values()) {
      instance.texture.dispose();
    }
    this.lodInstances.clear();

    if (this.sortWorker) {
      this.sortWorker.dispose();
      this.sortWorker = null;
    }
    for (const worker of this.sortPool) {
      worker.dispose();
    }
    this.sortPool = [];
    this.parallelSortBuffers = undefined;
    if (this.lodWorker) {
      this.lodWorker.dispose();
      this.lodWorker = null;
    }
    if (this.pickWorker) {
      this.pickWorker.dispose();
      this.pickWorker = null;
    }
    if (this.pager) {
      this.pager.dispose();
      this.pager = undefined;
    }
  }

  // dispose() rejects the work it interrupts.
  private dropErrorIfDisposed(error: unknown) {
    if (!this.isDisposed) {
      throw error;
    }
  }

  setDirty() {
    if (!this.isDisposed && !this.dirty) {
      this.dirty = true;
      this.onDirty?.();
    }
  }

  onBeforeRender(
    renderer: SparkHostRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
  ): void | Promise<void> {
    if (this.isDisposed) return;
    if (this.native) {
      if (this.autoUpdate)
        return this.updateInternal({ scene, camera, autoUpdate: true });
      return;
    }
    if (!isWebGLRenderer(renderer))
      throw new Error("Spark: expected the configured WebGL renderer");
    const spark = SparkRenderer.sparkOverride ?? this;

    const frame = renderer.info.render.frame;
    const isNewFrame = frame !== spark.lastFrame;
    spark.lastFrame = frame;

    // Trigger update (either sync in case of preUpdate, or async through setTimeout)
    // Skipped during a pickAsync() render: that pass reuses the splat
    // ordering/LoD selection already computed for the current frame (just a
    // narrower projection), so it must not kick off an extra sort/LoD update.
    if (spark.autoUpdate && isNewFrame && !spark.pickPass) {
      const preUpdate = spark.preUpdate && !renderer.xr.isPresenting;
      const useCamera = renderer.xr.isPresenting
        ? renderer.xr.getCamera()
        : camera;
      if (preUpdate) {
        spark
          .updateInternal({
            scene,
            camera: useCamera,
            autoUpdate: true,
          })
          .catch((error) => spark.dropErrorIfDisposed(error));
      } else {
        if (spark.updateTimeoutId === undefined) {
          spark.updateTimeoutId = setTimeout(() => {
            spark.updateTimeoutId = undefined;
            spark
              .updateInternal({
                scene,
                camera: useCamera,
                autoUpdate: true,
              })
              .catch((error) => spark.dropErrorIfDisposed(error));
          }, 1);
        }
      }
    }

    // Determine render target
    const currentRenderTarget = renderer.getRenderTarget();
    const isXRRenderTarget = checkIsXRRenderTarget(currentRenderTarget);
    // Objects like Reflector reset the renderer's current viewport, so prefer
    // the camera's own viewport.
    const viewport =
      camera.viewport ?? renderer.getCurrentViewport(new THREE.Vector4());
    if (viewport.z > 1 && viewport.w > 1) {
      spark.renderSize.set(viewport.z, viewport.w);
    } else if (currentRenderTarget) {
      spark.renderSize.set(
        currentRenderTarget.width,
        currentRenderTarget.height,
      );

      // WebXR mode on Apple Vision Pro returns 1x1 when presenting.
      // Use a different means to figure out the render size.
      if (
        isXRRenderTarget &&
        spark.renderSize.x === 1 &&
        spark.renderSize.y === 1
      ) {
        const baseLayer = renderer.xr.getSession()?.renderState.baseLayer;
        if (baseLayer) {
          spark.renderSize.x = baseLayer.framebufferWidth;
          spark.renderSize.y = baseLayer.framebufferHeight;
        }
      }
    } else {
      renderer.getDrawingBufferSize(spark.renderSize);
    }
    this.uniforms.renderSize.value.copy(spark.renderSize);

    const typedCamera = camera as
      | THREE.PerspectiveCamera
      | THREE.OrthographicCamera;

    this.uniforms.near.value = typedCamera.near;
    this.uniforms.far.value = typedCamera.far;

    const geometry = this.geometry as SplatGeometry;
    geometry.instanceCount = spark.activeSplats;
    this.uniforms.instanceCount.value = spark.activeSplats;

    // Stochastic mode: decided only on canvas/XR frames, never during a pick
    // pass (this.material is then a pick material) or a render to another
    // render target (captures, probes), which keep the sorted path.
    const isCanvasRender =
      !spark.pickPass && (currentRenderTarget === null || isXRRenderTarget);
    if (isCanvasRender && isNewFrame) {
      spark.updateStochastic(camera);
    }
    const stochastic = isCanvasRender && spark.stochasticActive;
    this.uniforms.stochastic.value = stochastic;
    if (stochastic) {
      // Blending is GL state only: switching it (and depthWrite/depthTest)
      // reuses the same program. Toggling `transparent` instead would flip
      // three's OPAQUE define and switch programs, and move the mesh between
      // render lists. Restored in onAfterRender so the material keeps its
      // configured state for every other draw and for pick materials.
      const material = this.material;
      this.stochasticSaved = {
        blending: material.blending,
        depthWrite: material.depthWrite,
        depthTest: material.depthTest,
      };
      material.blending = THREE.NoBlending;
      material.depthWrite = true;
      material.depthTest = true;
    }

    const accumToWorld = new THREE.Matrix4();
    if (!spark.display.extSplats) {
      accumToWorld.makeTranslation(spark.display.viewOrigin);
    }
    const cameraToWorld = camera.matrixWorld.clone();
    const worldToCamera = cameraToWorld.invert();
    const accumToCamera = worldToCamera.multiply(accumToWorld);
    accumToCamera.decompose(
      this.uniforms.renderToViewPos.value,
      this.uniforms.renderToViewQuat.value,
      new THREE.Vector3(),
    );
    this.uniforms.renderToViewBasis.value.setFromMatrix4(accumToCamera);

    this.uniforms.maxStdDev.value = spark.maxStdDev;
    this.uniforms.minPixelRadius.value = spark.minPixelRadius;
    this.uniforms.maxPixelRadius.value = spark.maxPixelRadius;
    this.uniforms.minAlpha.value = spark.minAlpha;
    this.uniforms.enable2DGS.value = spark.enable2DGS;
    // this.uniforms.enableRayEval.value = spark.enableRayEval;
    this.uniforms.lodInflate.value = spark.lodInflate;
    this.uniforms.preBlurAmount.value = spark.preBlurAmount;
    this.uniforms.blurAmount.value = spark.blurAmount;
    this.uniforms.focalDistance.value = spark.focalDistance;
    this.uniforms.apertureAngle.value = spark.apertureAngle;
    this.uniforms.falloff.value = spark.falloff;
    this.uniforms.clipXY.value = spark.clipXY;
    this.uniforms.focalAdjustment.value = spark.focalAdjustment;

    const outputColorSpace =
      currentRenderTarget === null
        ? renderer.outputColorSpace
        : isXRRenderTarget
          ? currentRenderTarget.texture.colorSpace
          : THREE.ColorManagement.workingColorSpace;
    this.uniforms.encodeLinear.value =
      outputColorSpace !== THREE.SRGBColorSpace;

    this.uniforms.ordering.value =
      spark.orderingTexture ?? SparkRenderer.emptyOrdering;
    this.uniforms.enableExtSplats.value = spark.display.extSplats;
    this.uniforms.enableCovSplats.value = spark.display.covSplats;
    if (spark.display.extSplats) {
      const extSplats = spark.display.getTextures();
      this.uniforms.extSplats.value = extSplats[0];
      this.uniforms.extSplats2.value = extSplats[1];
    } else {
      const packedSplats = spark.display.getTextures();
      this.uniforms.extSplats.value = packedSplats[0];
      this.uniforms.extSplats2.value = packedSplats[0];
    }

    this.uniforms.time.value = spark.display.time;
    this.uniforms.deltaTime.value = spark.display.deltaTime;
    // Alternating debug flag that can aid in visual debugging
    this.uniforms.debugFlag.value = (performance.now() / 1000.0) % 2.0 < 1.0;

    spark.dirty = false;
  }

  private readonly orderingTrace = new OrderingTrace();

  onAfterRender(
    _renderer: SparkHostRenderer,
    _scene: THREE.Scene,
    camera: THREE.Camera,
  ): void {
    const saved = this.stochasticSaved;
    if (saved) {
      this.stochasticSaved = null;
      this.material.blending = saved.blending;
      this.material.depthWrite = saved.depthWrite;
      this.material.depthTest = saved.depthTest;
    }
    const spark = SparkRenderer.sparkOverride ?? this;
    if (!spark.native && !spark.isDisposed) {
      spark.orderingTrace.draw(
        spark.display.mappingVersion,
        camera,
        spark.activeSplats,
      );
    }
  }

  /** @internal Also called by the native WebGPU renderer per canvas frame. */
  updateStochastic(camera: THREE.Camera) {
    const now = performance.now();
    const pose = this.stochasticPose;
    const world = camera.matrixWorld.elements;
    const proj = camera.projectionMatrix.elements;
    let moved = false;
    for (let i = 0; i < 16; ++i) {
      if (
        Math.abs(pose[i] - world[i]) > STOCHASTIC_POSE_EPSILON ||
        Math.abs(pose[16 + i] - proj[i]) > STOCHASTIC_POSE_EPSILON
      ) {
        moved = true;
        break;
      }
    }
    if (moved) {
      pose.set(world, 0);
      pose.set(proj, 16);
      this.stochasticMoveTime = now;
    }

    this.stochasticActive =
      this.stochastic === "on" ||
      (this.stochastic === "movement" &&
        now - this.stochasticMoveTime < this.stochasticSettleMs);

    // Moving in "movement" mode: (re)arm a timer so on-demand renderers
    // draw the clean sorted frame once the camera has been still long enough.
    if (this.stochasticActive && this.stochastic === "movement" && moved) {
      clearTimeout(this.stochasticTimeoutId);
      this.stochasticTimeoutId = setTimeout(() => {
        this.stochasticTimeoutId = undefined;
        this.setDirty();
      }, this.stochasticSettleMs + 1);
    }
  }

  clearSplats() {
    this.activeSplats = 0;
    this.display.numSplats = 0;
    this.setDirty();
  }

  async update({
    scene,
    camera,
  }: {
    scene: THREE.Scene;
    camera: THREE.Camera;
  }) {
    await this.updateInternal({ scene, camera, autoUpdate: false });
  }

  // /**
  //  * Provide additional cameras to prefetch paged splat chunks without
  //  * affecting main LOD selection.
  //  */
  // setPrefetchCameras(cameras?: THREE.Camera[], lodScaleMultiplier = 1.0) {
  //   const next = cameras?.filter(Boolean) ?? [];
  //   const sameCameras =
  //     this.prefetchCameras.length === next.length &&
  //     this.prefetchCameras.every((camera, index) => camera === next[index]);
  //   if (sameCameras && this.prefetchLodScale === lodScaleMultiplier) {
  //     return;
  //   }
  //   this.prefetchCameras = next;
  //   this.prefetchLodScale = lodScaleMultiplier;
  //   this.invalidatePrefetchCache();
  // }

  // /**
  //  * Invalidate the prefetch meshes cache. Call this when SplatMeshes are
  //  * added or removed from the scene.
  //  */
  // invalidatePrefetchCache() {
  //   this.prefetchMeshesCacheScene = undefined;
  // }

  private async updateInternal({
    scene,
    camera,
    autoUpdate,
  }: {
    scene: THREE.Scene;
    camera: THREE.Camera;
    autoUpdate: boolean;
  }) {
    if (this.isDisposed) {
      return;
    }
    if (this.ownsTimer) {
      this.timer.update();
    }
    if (this.native) {
      const visibleGenerators = await this.native.update(scene, camera);
      if (this.enableDriveLod && !this.isDisposed)
        this.driveLod({ visibleGenerators, camera, scene });
      return;
    }
    const renderer = this.requireWebGLRenderer();

    const center = camera.getWorldPosition(new THREE.Vector3());
    const dir = camera.getWorldDirection(new THREE.Vector3());

    const sortCullCos =
      this.sortCullFov > 0 &&
      !(camera as THREE.OrthographicCamera).isOrthographicCamera
        ? Math.cos((Math.min(this.sortCullFov, 360) * Math.PI) / 360)
        : -2;

    // Compare against the pose of the last generated accumulator, not the
    // last started sort: while a sort is in flight the camera may already have
    // stopped at the pose this.current was generated at, so regenerating every
    // frame would be redundant. sortDirty stays set from that generate, so the
    // final sort still runs once the in-flight one completes.
    const positionChanged =
      center.distanceTo(this.current.viewOrigin) > 0.001 ||
      sortCullCos !== this.sortCullCos;
    const directionChanged = dir.dot(this.current.viewDirection) < 0.999;

    const next = this.accumulators.pop();
    if (!next) {
      // Should never happen
      throw new Error("No next accumulator");
    }
    if (next === this.current) {
      // Should never happen
      throw new Error(
        "Next accumulator is the same as the current accumulator",
      );
    }
    const { version, mappingVersion, visibleGenerators, generate } =
      next.prepareGenerate({
        renderer,
        scene,
        timer: this.timer,
        camera,
        sortRadial: this.sortRadial ?? true,
        sortCullCos,
        renderSize: this.renderSize,
        previous: this.current,
        lodInstances: this.enableLod ? this.lodInstances : undefined,
      });

    // Meshes still loading contribute nothing this frame; request a render
    // when they finish so on-demand apps show them without other input.
    // Listen for the event rather than mesh.initialized: attaching to that
    // promise would mark a load failure as handled and silence its report.
    for (const generator of visibleGenerators) {
      if (
        generator instanceof SplatMesh &&
        !generator.isInitialized &&
        !this.initWatched.has(generator)
      ) {
        this.initWatched.add(generator);
        const onInitialized = () => {
          generator.removeEventListener("initialized", onInitialized);
          this.initWatched.delete(generator);
          this.setDirty();
        };
        generator.addEventListener("initialized", onInitialized);
      }
    }

    let doUpdate = true;
    // Only omit orientation invalidation for proven built-in generation.
    // prepareGenerate/frameUpdate above still observes changes on every update.
    let directionIndependent = false;
    if (
      (this.sortRadial ?? true) &&
      // Cone culling (sortCullFov) selects splats by view direction.
      sortCullCos === -2 &&
      directionChanged &&
      !positionChanged &&
      !renderer.xr.isPresenting
    ) {
      directionIndependent = visibleGenerators.every(
        (node) =>
          Object.getPrototypeOf(node) === SplatMesh.prototype &&
          node instanceof SplatMesh &&
          node.hasNativeSourceGenerator() &&
          !node.onFrame &&
          !node.paged &&
          !node.covSplats &&
          (Object.getPrototypeOf(node.splats) === PackedSplats.prototype ||
            Object.getPrototypeOf(node.splats) === ExtSplats.prototype) &&
          !node.objectModifiers?.length &&
          !node.worldModifiers?.length &&
          !node.covObjectModifiers?.length &&
          !node.covWorldModifiers?.length &&
          !node.skinning &&
          !node.edits?.length &&
          !node.rgbaDisplaceEdits &&
          !node.splatRgba,
      );
      scene.traverseVisible((node) => {
        if (isSplatEdit(node)) directionIndependent = false;
      });
    }
    const viewChanged =
      positionChanged || (directionChanged && !directionIndependent);
    const needsUpdate = viewChanged || version !== this.current.version;
    const mappingUpdated = mappingVersion !== this.display.mappingVersion;
    this.latestMappingVersion = mappingVersion;

    if (autoUpdate && !needsUpdate) {
      // Triggered by auto-update but no change
      doUpdate = false;
    }

    if (mappingUpdated && this.sorting) {
      // We need to be able to sort the splats because the mapping has changed.
      // Try again next time around.
      doUpdate = false;
    }

    if (!doUpdate) {
      // Restore unused accumulator to the free list
      this.accumulators.push(next);
    } else {
      generate();
      this.sortCullCos = sortCullCos;
      this.orderingTrace.generatedAt(next);

      if (this.flushAfterGenerate) {
        const gl = renderer.getContext() as WebGL2RenderingContext;
        gl.flush();
      }

      if (this.display.mappingVersion === next.mappingVersion) {
        // Same splat mapping so let's display it immediately and
        // reuse the sort order
        this.accumulators.push(this.display);
        this.display = next;
      } else {
        if (this.display !== this.current) {
          // The previous current is not being displayed, so replace it
          this.accumulators.push(this.current);
        }
      }

      this.current = next;
      this.sortDirty = true;
      this.setDirty();
    }

    if (this.enableDriveLod) {
      this.driveLod({ visibleGenerators, camera, scene });
    }
    await this.driveSort();
  }

  private async driveSort() {
    if (
      this.isDisposed ||
      this.sorting ||
      !this.sortDirty ||
      this.latestMappingVersion !== this.current.mappingVersion ||
      this.current.mappingChanged()
    ) {
      return;
    }

    if (this.sortTimeoutId !== undefined) {
      clearTimeout(this.sortTimeoutId);
      this.sortTimeoutId = undefined;
    }

    const now = performance.now();
    const nextSortTime = this.lastSortTime
      ? this.lastSortTime + this.minSortIntervalMs
      : now;
    if (now < nextSortTime) {
      this.sortTimeoutId = setTimeout(() => {
        this.sortTimeoutId = undefined;
        this.driveSort().catch((error) => this.dropErrorIfDisposed(error));
      }, nextSortTime - now);
      return;
    }

    this.sorting = true;
    this.sortDirty = false;
    this.lastSortTime = now;

    try {
      if (this.readPause > 0) {
        await new Promise((resolve) => setTimeout(resolve, this.readPause));
      }

      if (this.isDisposed) {
        return;
      }

      const current = this.current;
      const traceRequest = this.orderingTrace.request(
        current,
        this.sortRadial ?? true,
      );

      this.sortedCenter.copy(current.viewOrigin);
      this.sortedDir.copy(current.viewDirection);

      const { numSplats, maxSplats } = current;
      const rows = Math.max(1, Math.ceil(maxSplats / 16384));
      const orderingMaxSplats = rows * 16384;
      this.maxSplats = Math.max(this.maxSplats, orderingMaxSplats);

      const readback = Readback.ensureBuffer(maxSplats, this.readback32);
      this.readback32 = readback;

      await this.readbackDepth({
        current,
        renderer: this.requireWebGLRenderer(),
        numSplats,
        readback,
      });

      if (this.sortPause > 0) {
        await new Promise((resolve) => setTimeout(resolve, this.sortPause));
      }

      if (this.isDisposed) {
        return;
      }
      const threads = Math.floor(this.sortThreads);
      let ordering: Uint32Array;
      let activeSplats: number;
      if (
        threads > 1 &&
        numSplats >= SORT_PARALLEL_MIN_SPLATS &&
        globalThis.crossOriginIsolated
      ) {
        // Read back into the regular buffer (readPixels into shared memory is
        // not portable), then one copy into the shared keys.
        const buffers = ensureParallelSortBuffers(
          this.parallelSortBuffers,
          this.maxSplats,
          threads,
        );
        this.parallelSortBuffers = buffers;
        buffers.keys.set(readback.subarray(0, numSplats));
        while (this.sortPool.length < threads) {
          this.sortPool.push(new SplatWorker());
        }
        const sortPool = this.sortPool;
        const parallelSort = this.sort16 ? parallelSort16 : parallelSort32;
        activeSplats = await parallelSort({
          numSplats,
          buffers,
          call: (w, name, args) => sortPool[w].call(name, args as never),
        });
        // WebGL may reject uploads from shared memory: copy to a regular array
        ordering = this.orderingBuffers.take(this.maxSplats);
        ordering.set(buffers.ordering.subarray(0, activeSplats));
      } else {
        if (!this.sortWorker) {
          this.sortWorker = new SplatWorker();
        }
        const result = await this.sortWorker.call(
          this.sort16 ? "sortSplats32as16" : "sortSplats32",
          {
            numSplats,
            readback,
            ordering: this.orderingBuffers.take(this.maxSplats),
          },
        );
        this.readback32 = result.readback;
        ordering = result.ordering;
        activeSplats = result.activeSplats;
      }

      const traceCompletedAt = this.orderingTrace.complete(traceRequest);

      if (this.sortDelay > 0) {
        await new Promise((resolve) => setTimeout(resolve, this.sortDelay));
      }

      if (this.isDisposed) {
        return;
      }

      this.activeSplats = activeSplats;

      if (this.orderingTexture) {
        if (rows > this.orderingTexture.image.height) {
          this.orderingTexture.dispose();
          this.orderingTexture = null;
        }
      }

      if (!this.orderingTexture) {
        // console.log(`Allocating orderingTexture: ${4096}x${rows}`);
        const orderingTexture = new THREE.DataTexture(
          ordering,
          4096,
          rows,
          THREE.RGBAIntegerFormat,
          THREE.UnsignedIntType,
        );
        orderingTexture.internalFormat = "RGBA32UI";
        orderingTexture.needsUpdate = true;
        this.orderingBuffers.attach(orderingTexture, ordering);
        this.orderingTexture = orderingTexture;
      } else {
        this.orderingBuffers.attach(this.orderingTexture, ordering);
        const renderer = this.requireWebGLRenderer();
        if (!renderer.properties.has(this.orderingTexture)) {
          this.orderingTexture.needsUpdate = true;
        } else {
          // Only rows holding the first activeSplats entries are ever read
          // (instanceCount = activeSplats, 4096 RGBA texels = 16384 per row).
          uploadU32DataTextureRows(
            renderer,
            this.orderingTexture,
            4096,
            Math.max(1, Math.ceil(activeSplats / 16384)),
            ordering,
          );
        }
      }

      // console.log(`Sorted (${this.minSortIntervalMs}) ${numSplats} splats in ${(performance.now() - now).toFixed(0)} ms`);

      this.orderingTrace.commit(traceRequest, traceCompletedAt, activeSplats);

      if (this.current.mappingVersion === current.mappingVersion) {
        if (this.current.mappingVersion !== this.display.mappingVersion) {
          this.accumulators.push(this.display);
          this.display = this.current;
        }
      }
      this.setDirty();
    } catch (error) {
      // Disposing the worker rejects its outstanding call. That cancellation
      // must not revive resources or escape a scheduled background sort.
      if (this.isDisposed) return;
      this.sortDirty = true;
      throw error;
    } finally {
      this.sorting = false;
    }
    if (!this.isDisposed) {
      this.driveSort().catch((error) => this.dropErrorIfDisposed(error));
    }
  }

  private ensureLodWorker() {
    if (this.disposed) {
      throw new DOMException("SparkRenderer disposed", "AbortError");
    }
    if (!this.lodWorker) {
      this.lodWorker = new SplatWorker();
      this.pickWorker = new SplatWorker();
      // Queued before any tree message: only the pick worker computes radii
      this.callPickWorker("setLodRadii", { enabled: true });
    }
    return this.lodWorker;
  }

  /**
   * Mirror a LoD tree message to the pick worker. Call it right before the
   * matching lodWorker call, in the same tick, with copies of any buffers
   * (lodWorker takes ownership of its own), so the two FIFOs stay in step.
   */
  private callPickWorker<T extends keyof rpcHandlers>(
    name: T,
    args: Parameters<rpcHandlers[T]>[0],
    lodId?: Promise<{ lodId: number }>,
  ) {
    const pick = this.pickWorker?.call(name, args);
    if (!pick) return;
    Promise.all([pick, lodId]).then(
      ([pickResult, lodResult]) => {
        const pickLodId = (pickResult as { lodId?: number } | undefined)?.lodId;
        if (lodResult && pickLodId !== lodResult.lodId) {
          console.error(
            `pick worker lodId ${pickLodId} != ${lodResult.lodId} (${name})`,
          );
        }
      },
      (error) => console.error(`pick worker ${name}:`, error),
    );
  }

  defaultSplatTarget() {
    return isOculus()
      ? 500000
      : isVisionPro()
        ? 750000
        : isAndroid()
          ? 1000000
          : isIos()
            ? 1500000
            : 2500000;
  }

  private driveLod({
    visibleGenerators,
    camera,
    scene,
  }: {
    visibleGenerators: SplatGenerator[];
    camera: THREE.Camera;
    scene: THREE.Scene;
  }) {
    if (this.disposed) return;
    const defaultSplatCount = this.defaultSplatTarget();
    const splatCount = this.lodSplatCount ?? defaultSplatCount;
    const maxSplats = splatCount * this.lodSplatScale;

    let pixelScaleLimit = 0.0;
    let fovScale = 1;
    if (camera instanceof THREE.PerspectiveCamera) {
      // three.js leaves fov at its default on the WebXR camera
      const tanYfov = 1.0 / camera.projectionMatrix.elements[5];
      pixelScaleLimit = (2.0 * tanYfov) / this.renderSize.y;
      fovScale = lodMetaFovScale(tanYfov, camera.aspect);
    } else if (camera instanceof THREE.OrthographicCamera) {
      // Effective visible size after zoom
      const viewHeight = (camera.top - camera.bottom) / camera.zoom;
      const viewWidth = (camera.right - camera.left) / camera.zoom;
      // World/view units per pixel (constant with depth for ortho)
      const pxY = viewHeight / Math.max(1, this.renderSize.y);
      const pxX = viewWidth / Math.max(1, this.renderSize.x);
      pixelScaleLimit = Math.min(pxX, pxY);
    }

    pixelScaleLimit *= this.lodRenderScale;

    const viewPos = new THREE.Vector3();
    const viewQuat = new THREE.Quaternion();
    this.current.viewToWorld.decompose(viewPos, viewQuat, new THREE.Vector3());

    if (this.lodPosOverride) {
      viewPos.copy(this.lodPosOverride);
    }
    if (this.lodQuatOverride) {
      viewQuat.copy(this.lodQuatOverride).normalize();
    }

    let cameraMoved = false;
    if (this.lastLod) {
      if (this.lastLod.maxSplats !== maxSplats) {
        this.lodDirty = true;
      }
      if (
        Math.abs(this.lastLod.pixelScaleLimit - pixelScaleLimit) >
        1e-5 * pixelScaleLimit
      ) {
        this.lodDirty = true;
      }
      if (this.lastLod.maxSplats !== maxSplats) {
        this.lodMetaDirty = true;
      }

      const distance = viewPos.distanceTo(this.lastLod.pos);
      const distanceRamp = Math.max(0.0, 1.0 - distance / 1.0);
      const dot = viewQuat.dot(this.lastLod.quat);
      const quatRamp = Math.max(0.0, 1.0 - (1.0 - dot) / 0.01);
      const similarity = distanceRamp * quatRamp;
      cameraMoved = similarity < 0.999;
    }

    const lodMeshes = !this.enableLod
      ? []
      : (visibleGenerators.filter((generator) => {
          return (
            generator instanceof SplatMesh &&
            (generator.packedSplats?.lodSplats ||
              generator.extSplats?.lodSplats ||
              generator.paged) &&
            generator.enableLod !== false
          );
        }) as SplatMesh[]);
    const hasPaged = lodMeshes.some((mesh) => mesh.paged);
    const hasLodMeta = lodMeshes.some((mesh) => mesh.paged?.lodMetaPromise);

    // Movimento qualsiasi: come prima per le mesh normali; se ci sono solo mesh
    // lod-meta decide il gate PlayCanvas sotto
    if (
      cameraMoved &&
      (!hasLodMeta || lodMeshes.some((mesh) => !mesh.paged?.lodMetaPromise))
    ) {
      this.lodDirty = true;
    }

    if (hasLodMeta) {
      this.lodMetaFovScale = fovScale;
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(viewQuat);
      const last = this.lodMetaLast;
      const gateMoved =
        !last ||
        viewPos.distanceTo(last.pos) > 0.5 ||
        fwd.dot(last.fwd) < Math.cos(Math.PI / 2) ||
        Math.abs(fovScale - last.fovScale) > last.fovScale * 0.02;
      if (++this.lodMetaFrames >= 10) {
        this.lodMetaFrames = 0;
        if (gateMoved) {
          this.lodMetaLast = { pos: viewPos.clone(), fwd, fovScale };
          this.lodMetaDirty = true;
          this.lodDirty = true;
        }
      } else if (gateMoved) {
        // ponytail: render on-demand, assicura il frame del prossimo controllo
        this.setDirty();
      }
    }

    if (this.lodMeshes.length !== lodMeshes.length) {
      this.lodDirty = true;
      this.lodMetaDirty = true;
    } else {
      if (
        lodMeshes.some(
          (m, i) =>
            m !== this.lodMeshes[i].mesh ||
            m.version - m.introDotsVersion > this.lodMeshes[i].version,
        )
      ) {
        this.lodDirty = true;
        this.lodMetaDirty = true;
      }
    }

    // L'animazione introDots rigenera gli splat ma non riseleziona il LoD
    this.lodMeshes = lodMeshes.map((mesh) => ({
      mesh,
      version: mesh.version - mesh.introDotsVersion + 1,
    }));

    // if (!this.lodDirty && lodMeshes.length === 0 && this.lodIds.size === 0) {
    //   return;
    // }

    this.lodInitQueue = [];
    const now = performance.now();

    for (const mesh of lodMeshes) {
      const splats =
        mesh.packedSplats?.lodSplats ?? mesh.extSplats?.lodSplats ?? mesh.paged;
      if (splats) {
        const record = this.lodIds.get(splats);
        if (record) {
          record.lastTouched = now;
        } else if (!this.lodInitFailed.has(splats)) {
          this.lodInitQueue.push(splats);
        }
      }
    }

    this.ensureLodWorker().tryExclusive(async (worker) => {
      try {
        await this.driveLodExclusive(worker, {
          hasPaged,
          lodMeshes,
          viewPos,
          viewQuat,
          pixelScaleLimit,
          maxSplats,
        });
      } catch (error) {
        this.dropErrorIfDisposed(error);
      } finally {
        // driveLod() skips tryExclusive while this runs, so work it flagged
        // (lodDirty, lodInitQueue) would otherwise wait for an unrelated render.
        // Likewise chunk data that landed after consumeLodTreeUpdates: its
        // onUpdate render already happened and found the worker busy.
        if (
          this.lodDirty ||
          this.lodInitQueue.length > 0 ||
          this.pager?.hasQueued()
        ) {
          this.setDirty();
        }
      }
    });
  }

  /** Body of the LoD update, run with exclusive access to the LoD worker. */
  private async driveLodExclusive(
    worker: SplatWorker,
    {
      hasPaged,
      lodMeshes,
      viewPos,
      viewQuat,
      pixelScaleLimit,
      maxSplats,
    }: {
      hasPaged: boolean;
      lodMeshes: SplatMesh[];
      viewPos: THREE.Vector3;
      viewQuat: THREE.Quaternion;
      pixelScaleLimit: number;
      maxSplats: number;
    },
  ) {
    if (hasPaged && !this.pager) {
      this.pager = new SplatPager({
        renderer: this.native ? null : this.requireWebGLRenderer(),
        extSplats: this.pagedExtSplats,
        maxSplats: this.maxPagedSplats,
        numFetchers: this.numLodFetchers,
        onUpdate: () => this.setDirty(),
      });

      const args = { capacity: this.pager.maxSplats };
      const call = worker.call("newLodTree", args);
      this.callPickWorker("newLodTree", args, call);
      const { lodId } = await call;
      this.pagerId = lodId;
    }

    // Assign pager to any new meshes that don't have one yet
    // (must run every frame, not just when pager is first created)
    if (this.pager) {
      for (const { mesh } of this.lodMeshes) {
        if (mesh.paged && !mesh.paged.pager) {
          mesh.paged.pager = this.pager;
        }
      }
    }

    if (this.lodInitQueue.length > 0) {
      const lodInitQueue = this.lodInitQueue;
      this.lodInitQueue = [];
      while (lodInitQueue.length > 0) {
        const splats = lodInitQueue.shift();
        if (splats) {
          try {
            await this.initLodTree(worker, splats);
            this.lodDirty = true;
          } catch (error) {
            if (this.isDisposed) {
              return;
            }
            this.lodInitFailed.add(splats);
            console.error("initLodTree error", error);
          }
        }
      }
    }

    if (this.pager) {
      const updates = this.pager.consumeLodTreeUpdates();

      for (const { splats, page, chunk, numSplats, lodTree } of updates) {
        const record = this.lodIds.get(splats);
        if (splats.lodMetaPromise) {
          // lod-meta.json: niente albero Rust, basta rifare la selezione
          // (file arrivato/rimosso: sempre, fuori dal gate camera)
          this.lodDirty = true;
          this.lodMetaDirty = true;
        } else if (record) {
          if (lodTree && chunk === 0) {
            record.rootPage = page;
          } else if (!lodTree && chunk === 0 && record.rootPage === page) {
            // Root chunk evicted: forget the page (traversal skips this mesh
            // until the root is refetched) and stop drawing indices into
            // pages that no longer hold this mesh's data.
            record.rootPage = undefined;
            splats.clear();
          }
          this.lodUpdates.push({
            lodId: record.lodId,
            pageBase: page * this.pager.pageSplats,
            chunkBase: chunk * this.pager.pageSplats,
            count: numSplats,
            lodTreeData: lodTree,
          });
        }
      }
    }

    if (this.lodUpdates.length > 0) {
      const lodUpdates = this.lodUpdates;
      this.lodUpdates = [];
      this.callPickWorker("updateLodTrees", {
        ranges: lodUpdates.map((range) => ({
          ...range,
          lodTreeData: range.lodTreeData?.slice(),
        })),
      });
      await worker.call("updateLodTrees", { ranges: lodUpdates });
      this.lodPagesPending = true;
    }

    if (this.lodPagesPending && !this.lodDirty) {
      const wait = this.lastPagesLodTime + 300 - performance.now();
      if (wait <= 0) {
        this.lodDirty = true;
        this.lastPagesLodTime = performance.now();
      } else if (this.lodPagesTimer === undefined) {
        this.lodPagesTimer = setTimeout(() => {
          this.lodPagesTimer = undefined;
          this.setDirty();
        }, wait);
      }
    }

    if (this.lodDirty) {
      this.lodPagesPending = false;
      const now = performance.now();
      const deltaPred = new THREE.Vector3();
      if (this.lastLod) {
        const deltaTime = Math.max(1, now - this.lastLod.timestamp);
        deltaPred
          .copy(viewPos)
          .sub(this.lastLod.pos)
          .multiplyScalar(this.lastTraverseTime / deltaTime);
      }
      this.lastLod = {
        pos: viewPos,
        quat: viewQuat,
        pixelScaleLimit,
        maxSplats,
        timestamp: now,
      };
      this.lodDirty = false;

      await this.updateLodInstances(
        worker,
        deltaPred,
        lodMeshes,
        maxSplats,
        viewPos,
        viewQuat,
        pixelScaleLimit,
      );
      this.currentLod = this.lastLod;
      this.setDirty();
    }

    if (SPARK_ENABLE_HOOKS) {
      const p = sparkHook("lod.beforeCleanup", { spark: this });
      if (p) await p;
    }
    await this.cleanupLodTrees(worker);
  }

  private async initLodTree(
    worker: SplatWorker,
    splats: PackedSplats | ExtSplats | PagedSplats,
  ) {
    if (splats instanceof PackedSplats || splats instanceof ExtSplats) {
      const lodTree = splats.extra.lodTree as Uint32Array;
      const numSplats = splats.numSplats ?? 0;
      const call = worker.call("initLodTree", {
        numSplats,
        lodTree: lodTree.slice(),
      });
      this.callPickWorker(
        "initLodTree",
        { numSplats, lodTree: lodTree.slice() },
        call,
      );
      const { lodId } = await call;
      this.lodIds.set(splats, { lodId, lastTouched: performance.now() });
      this.lodIdToSplats.set(lodId, splats);
      // console.log("*** initLodTree", lodId, splats.extra.lodTree, splats);
    } else {
      const args = { lodId: this.pagerId };
      const call = worker.call("newSharedLodTree", args);
      this.callPickWorker("newSharedLodTree", args, call);
      const { lodId } = await call;
      this.lodIds.set(splats, { lodId, lastTouched: performance.now() });
      this.lodIdToSplats.set(lodId, splats);
      // console.log("*** newSharedLodTree", lodId, this.pagerId, splats);
    }
  }

  private pageSizeWarning = false;

  private async updateLodInstances(
    worker: SplatWorker,
    deltaPred: THREE.Vector3,
    lodMeshes: SplatMesh[],
    maxSplats: number,
    viewPos: THREE.Vector3,
    viewQuat: THREE.Quaternion,
    pixelScaleLimit: number,
  ) {
    // Commented out because it makes LoDing less stable
    // viewPos.add(deltaPred);

    const uuidToMesh: Map<string, SplatMesh> = new Map();
    const cameraToWorld = new THREE.Matrix4().compose(
      viewPos,
      viewQuat,
      new THREE.Vector3().setScalar(1),
    );

    // lod-meta.json (PlayCanvas streamed SOG): selezione per foglia in JS
    // (LodMeta.selectLodMeta), fuori dal traversal Rust. Budget splat diviso per
    // numero di mesh: le lodmeta prendono la loro quota, il traversal il resto.
    const lodMetaMeshes = lodMeshes.filter(
      (mesh) => mesh.paged?.lodMetaPromise,
    );
    const lodMetaBudget = lodMeshes.length
      ? Math.floor((maxSplats * lodMetaMeshes.length) / lodMeshes.length)
      : 0;
    const radMaxSplats = maxSplats - lodMetaBudget;

    const instances = lodMeshes.reduce(
      (instances, mesh) => {
        uuidToMesh.set(mesh.uuid, mesh);
        const viewToObject = mesh.matrixWorld
          .clone()
          .invert()
          .multiply(cameraToWorld);

        const splats =
          mesh.packedSplats?.lodSplats ??
          mesh.extSplats?.lodSplats ??
          mesh.paged;
        if (!splats) {
          return instances;
        }
        const record = this.lodIds.get(splats);
        if (!record) {
          return instances;
        }

        if (
          this.pager &&
          mesh.paged &&
          (record.rootPage === undefined || mesh.paged.lodMetaPromise)
        ) {
          return instances;
        }

        instances[mesh.uuid] = {
          instanceId: mesh.uuid,
          lodId: record.lodId,
          rootPage: record.rootPage,
          viewToObjectCols: viewToObject.elements,
          lodScale: mesh.lodScale,
          behindFoveate: mesh.behindFoveate ?? this.behindFoveate,
          coneFov0: mesh.coneFov0 ?? this.coneFov0,
          coneFov: mesh.coneFov ?? this.coneFov,
          coneFoveate: mesh.coneFoveate ?? this.coneFoveate,
        };
        return instances;
      },
      {} as Record<
        string,
        {
          instanceId: string;
          lodId: number;
          rootPage?: number;
          viewToObjectCols: number[];
          lodScale: number;
          behindFoveate: number;
          coneFov0: number;
          coneFov: number;
          coneFoveate: number;
        }
      >,
    );

    // Fork: standard mode derives the raycast set from this same traversal
    const wantRaycast =
      !!this.lodRaycast &&
      this.lodTraverseMode !== "dynamic" &&
      performance.now() - this.lastLodRaycastTime >= this.lodRaycastIntervalMs;
    if (wantRaycast) {
      this.lastLodRaycastTime = performance.now();
    }
    const traverseStart = performance.now();
    const result = await worker.call("traverseLodTrees", {
      maxSplats: radMaxSplats,
      pixelScaleLimit,
      lastPixelLimit: this.lastPixelLimit,
      instances,
      traverseMode: this.lodTraverseMode,
      raycastMaxSplats: wantRaycast ? this.lodRaycast : undefined,
    });
    this.lastTraverseTime = performance.now() - traverseStart;

    const { keyIndices, raycastKeyIndices, chunks, pixelLimit } = result;
    this.lastPixelLimit = pixelLimit;
    const totalLodSplats = Object.values(keyIndices).reduce(
      (sum, { numSplats }) => sum + numSplats,
      0,
    );
    // console.log(
    //   `traverseLodTrees in ${this.lastTraverseTime} ms, pixelLimit=${pixelLimit}, totalLodSplats=${totalLodSplats}`,
    // );

    this.updateLodIndices(uuidToMesh, keyIndices);
    if (raycastKeyIndices) {
      for (const [uuid, countIndices] of Object.entries(raycastKeyIndices)) {
        (uuidToMesh.get(uuid) as SplatMesh).raycastIndices = countIndices;
      }
    }
    // console.log("chunks.length =", chunks.length);

    if (this.pager) {
      this.pager.processUploads();

      const pagedMeshes = lodMeshes
        .map((mesh) => {
          if (!mesh.paged || !this.pager || mesh.paged.lodMetaPromise) {
            return null;
          }
          const meshPosition = mesh.getWorldPosition(new THREE.Vector3());
          return {
            splats: mesh.paged,
            distance: meshPosition.distanceTo(viewPos),
          };
        })
        .filter((result) => result !== null);

      if (!this.pageSizeWarning && pagedMeshes.length > this.pager.maxPages) {
        this.pageSizeWarning = true;
        console.warn(
          `# paged SplatMeshes exceeds maxPages: ${pagedMeshes.length} > ${this.pager.maxPages}`,
        );
      }

      // Fetch root chunk of each paged splats in priority of distance to camera
      pagedMeshes.sort((a, b) => a.distance - b.distance);
      this.pager.fetchPriority = pagedMeshes.map(({ splats }) => ({
        splats,
        chunk: 0,
      }));

      const radChunks: { splats: PagedSplats; chunk: number }[] = [];
      for (const [lodId, chunk] of chunks) {
        const splats = this.lodIdToSplats.get(lodId);
        if (splats instanceof PagedSplats) {
          if (chunk !== 0) {
            radChunks.push({ splats, chunk });
          }
        }
      }

      if (lodMetaMeshes.length > 0 && !this.lodMetaDirty) {
        // gate PlayCanvas chiuso: selezione invariata, ripropone le stesse priorità
        this.pager.fetchPriority.push(...this.lodMetaPriority, ...radChunks);
        this.pager.fetchPriority.push(...this.lodMetaPrefetch);
      } else if (lodMetaMeshes.length > 0) {
        this.lodMetaDirty = false;
        const pager = this.pager;
        const ready = lodMetaMeshes.filter(
          (mesh) => mesh.paged?.lodMeta && mesh.paged.pager === pager,
        );
        if (
          lodMetaMeshes.some(
            (mesh) => !mesh.paged?.lodMeta && !mesh.paged?.lodMetaError,
          )
        ) {
          // lod-meta.json ancora in caricamento: riprova al prossimo giro
          this.lodDirty = true;
          this.lodMetaDirty = true;
        }
        const radPages = this.pager.fetchPriority.length + radChunks.length;
        const selection = selectLodMeta({
          instances: ready.map((mesh) => ({
            splats: mesh.paged as PagedSplats,
            data: mesh.paged?.lodMeta as LodMetaData,
            viewToObject: mesh.matrixWorld
              .clone()
              .invert()
              .multiply(cameraToWorld),
            lodScale: mesh.lodScale,
          })),
          maxSplats: lodMetaBudget,
          // ponytail: 3/4 delle pagine libere ai file voluti, il resto a fallback/prefetch
          maxFiles: Math.max(1, Math.floor(0.75 * (pager.maxPages - radPages))),
          residentPage: (splats, file) => {
            const pageLru = pager.getSplatsChunk(splats, file);
            return pageLru && pager.isPageUploaded(pageLru.page)
              ? pageLru.page
              : -1;
          },
          indicesCache: ready.map((mesh) => mesh.paged?.lodMetaIndices),
          fovScale: this.lodMetaFovScale,
          lodLimit: this.lodMetaLimit,
        });
        if (selection.rangeChanged) {
          // bootstrap finito (tutto il livello grossolano caricato): range completo
          this.lodDirty = true;
          this.lodMetaDirty = true;
        }
        ready.forEach((mesh, m) => {
          const paged = mesh.paged as PagedSplats;
          paged.lodMetaIndices = selection.indices[m];
          if (selection.numSplats[m] > 0) {
            paged.update(selection.numSplats[m], selection.indices[m]);
          } else {
            paged.clear();
          }
        });
        // Ordine: radici .rad, file lodmeta mostrati/voluti, chunk .rad, prefetch
        this.lodMetaPriority = selection.priority;
        this.lodMetaPrefetch = selection.prefetch;
        this.pager.fetchPriority.push(...selection.priority, ...radChunks);
        this.pager.fetchPriority.push(...selection.prefetch);
      } else {
        this.pager.fetchPriority.push(...radChunks);
      }

      this.pager.autoDrive = this.enableLodFetching;
      if (this.enableLodFetching) {
        this.pager.driveFetchers();
      }
    }

    if (
      this.lodRaycast &&
      this.lodTraverseMode === "dynamic" &&
      performance.now() - this.lastLodRaycastTime >= this.lodRaycastIntervalMs
    ) {
      this.lastLodRaycastTime = performance.now();
      const traverseStart = performance.now();
      const result = await worker.call("traverseLodTrees", {
        maxSplats: Math.min(this.lodRaycast, Math.round(totalLodSplats * 0.1)),
        pixelScaleLimit,
        instances,
        traverseMode: this.lodTraverseMode,
      });
      const raycastTraverseTime = performance.now() - traverseStart;

      const { keyIndices } = result;
      const totalRaycastSplats = Object.values(keyIndices).reduce(
        (sum, { numSplats }) => sum + numSplats,
        0,
      );
      for (const [uuid, countIndices] of Object.entries(keyIndices)) {
        const mesh = uuidToMesh.get(uuid) as SplatMesh;
        mesh.raycastIndices = countIndices;
        // console.log("Set raycast indices", uuid, countIndices.numSplats, countIndices.indices.length);
      }
      // console.log(`raycast traverse in ${raycastTraverseTime} ms, totalRaycastSplats=${totalRaycastSplats}`);
    }
  }

  private async cleanupLodTrees(worker: SplatWorker) {
    // Release every expired tree. Meshes in the current render set are never
    // candidates, regardless of lastTouched (which is stamped in the prelude,
    // before `now` below), so a timeout of 0 means "as soon as the mesh stops
    // being rendered".
    const rendered = new Set(
      this.lodMeshes.map(
        ({ mesh }) =>
          mesh.packedSplats?.lodSplats ??
          mesh.extSplats?.lodSplats ??
          mesh.paged,
      ),
    );
    const now = performance.now();
    const expired = [];
    for (const [splats, record] of this.lodIds.entries()) {
      if (rendered.has(splats)) continue;
      if (record.lastTouched <= now - this.lodCleanupTimeoutMs) {
        expired.push({ splats, lodId: record.lodId });
      }
    }

    // All bookkeeping happens synchronously, before any await: driveLod() may
    // run between awaits and must see a consistent lodIds. Only the worker
    // disposals, which nothing references anymore, are awaited afterwards.
    for (const { splats, lodId } of expired) {
      this.lodIds.delete(splats);
      this.lodIdToSplats.delete(lodId);

      for (const [mesh, instance] of this.lodInstances.entries()) {
        if (instance.lodId === lodId) {
          instance.texture.dispose();
          this.lodInstances.delete(mesh);
        }
      }

      if (splats instanceof PagedSplats) {
        this.pager?.removeSplats(splats);
      }
    }

    for (const { lodId } of expired) {
      this.callPickWorker("disposeLodTree", { lodId });
      await worker.call("disposeLodTree", { lodId });
    }
  }

  private updateLodIndices(
    uuidToMesh: Map<string, SplatMesh>,
    keyIndices: Record<
      string,
      { lodId: number; numSplats: number; indices: Uint32Array }
    >,
  ) {
    // console.log("updateLodIndices", keyIndices);
    for (const [uuid, countIndices] of Object.entries(keyIndices)) {
      const { lodId, numSplats, indices } = countIndices;
      const mesh = uuidToMesh.get(uuid) as SplatMesh;

      if (mesh.paged) {
        mesh.paged.update(numSplats, indices);
        // console.log("*** paged.update", lodId, numSplats, indices.slice(0, 5).join(","));
      } else {
        let instance = this.lodInstances.get(mesh);
        if (instance) {
          if (indices.length > instance.indices.length) {
            instance.texture.dispose();
            instance = undefined;
          }
        }

        const rows = Math.ceil(indices.length / 16384);
        if (!instance) {
          const capacity = rows * 16384;
          if (indices.length !== capacity) {
            throw new Error("Indices length != capacity");
          }
          const texture = new THREE.DataTexture(
            indices,
            4096,
            rows,
            THREE.RGBAIntegerFormat,
            THREE.UnsignedIntType,
          );
          texture.internalFormat = "RGBA32UI";
          texture.needsUpdate = true;
          instance = { lodId, numSplats, indices, texture };
          this.lodInstances.set(mesh, instance);
        } else {
          instance.numSplats = numSplats;
          // instance.indices.set(indices.subarray(0, numSplats));

          if (this.native) {
            instance.indices = indices;
            instance.texture.image.data = indices;
          } else {
            const renderer = this.requireWebGLRenderer();
            if (renderer.properties.has(instance.texture)) {
              uploadU32DataTextureRows(
                renderer,
                instance.texture,
                4096,
                rows,
                indices,
              );
            }
          }
        }
      }
      mesh.updateMappingVersion();
    }
  }

  private async readbackDepth({
    current,
    renderer,
    numSplats,
    readback,
  }: {
    current: SplatAccumulator;
    renderer: THREE.WebGLRenderer;
    numSplats: number;
    readback: Uint32Array;
  }) {
    if (!renderer) {
      throw new Error("No renderer");
    }
    if (!current.target) {
      throw new Error("No target");
    }

    const roundedCount =
      Math.ceil(numSplats / SPLAT_TEX_WIDTH) * SPLAT_TEX_WIDTH;
    if (readback.byteLength < roundedCount * 4) {
      throw new Error(
        `Readback buffer too small: ${readback.byteLength} < ${roundedCount * 4}`,
      );
    }
    const readbackUint8 = new Uint8Array(readback.buffer);
    const renderState = this.saveRenderState(renderer);

    // We can only read back one 2D array layer of pixels at a time,
    // so loop through them, initiate the readback, and collect the
    // completion promises.
    const layerSize = SPLAT_TEX_WIDTH * SPLAT_TEX_HEIGHT;
    let baseIndex = 0;
    const promises = [];

    while (baseIndex < numSplats) {
      const layer = Math.floor(baseIndex / layerSize);
      const layerBase = layer * layerSize;
      const layerYEnd = Math.min(
        SPLAT_TEX_HEIGHT,
        Math.ceil((numSplats - layerBase) / SPLAT_TEX_WIDTH),
      );

      // Compute the subarray that this layer of readback corresponds to
      const readbackSize = SPLAT_TEX_WIDTH * layerYEnd * 4;
      const subReadback = readbackUint8.subarray(
        layerBase * 4,
        layerBase * 4 + readbackSize,
      );
      renderer.setRenderTarget(current.target, layer);

      const promise = renderer.readRenderTargetPixelsAsync(
        current.target,
        0,
        0,
        SPLAT_TEX_WIDTH,
        layerYEnd,
        subReadback,
        undefined,
        current.extSplats ? 2 : 1,
      );
      promises.push(promise);

      if (this.flushAfterRead) {
        const gl = renderer.getContext() as WebGL2RenderingContext;
        gl.flush();
      }

      baseIndex += SPLAT_TEX_WIDTH * layerYEnd;
    }

    this.resetRenderState(renderer, renderState);
    return Promise.all(promises).then(() => readback);
  }

  private saveRenderState(renderer: THREE.WebGLRenderer) {
    return {
      target: renderer.getRenderTarget(),
      xrEnabled: renderer.xr.enabled,
      autoClear: renderer.autoClear,
    };
  }

  private resetRenderState(
    renderer: THREE.WebGLRenderer,
    state: {
      target: THREE.WebGLRenderTarget | null;
      xrEnabled: boolean;
      autoClear: boolean;
    },
  ) {
    renderer.setRenderTarget(state.target);
    renderer.xr.enabled = state.xrEnabled;
    renderer.autoClear = state.autoClear;
  }

  private static emptyOrdering = (() => {
    const numIndices = 4 * 4096 * 1;
    const emptyArray = new Uint32Array(numIndices);
    const texture = new THREE.DataTexture(emptyArray, 4096, 1);
    texture.format = THREE.RGBAIntegerFormat;
    texture.type = THREE.UnsignedIntType;
    texture.internalFormat = "RGBA32UI";
    texture.needsUpdate = true;
    return texture;
  })();

  /**
   * Native WebGPU resolves after update and draw submission. Classic WebGL keeps
   * its existing background updates. Neither route waits for GPU completion.
   * `target` renders the full composite (Three.js objects and splats, with
   * depth) into that render target instead of the canvas.
   */
  async renderAsync(
    scene: THREE.Scene,
    camera: THREE.Camera,
    { target = null }: { target?: THREE.RenderTarget | null } = {},
  ): Promise<void> {
    if (this.native) return this.native.renderAsync(scene, camera, { target });
    if (!target) return this.render(scene, camera);
    const renderer = this.requireWebGLRenderer();
    const previous = renderer.getRenderTarget();
    try {
      renderer.setRenderTarget(target as THREE.WebGLRenderTarget);
      this.render(scene, camera);
    } finally {
      renderer.setRenderTarget(previous);
    }
  }

  render(scene: THREE.Scene, camera: THREE.Camera): void {
    if (this.isDisposed) throw new Error("SparkRenderer is disposed");
    if (this.native)
      throw new Error(
        "Spark: use await renderAsync(scene, camera) for native WebGPU",
      );
    try {
      SparkRenderer.sparkOverride = this;
      this.renderer.render(scene, camera);
    } finally {
      SparkRenderer.sparkOverride = undefined;
    }
  }

  renderTarget({
    scene,
    camera,
  }: { scene: THREE.Scene; camera: THREE.Camera }): THREE.WebGLRenderTarget {
    const target = this.backTarget ?? this.target;
    if (!target) {
      throw new Error("No target");
    }

    const previousTarget = this.requireWebGLRenderer().getRenderTarget();
    try {
      this.requireWebGLRenderer().setRenderTarget(target);
      SparkRenderer.sparkOverride = this;
      this.renderer.render(scene, camera);
    } finally {
      SparkRenderer.sparkOverride = undefined;
      this.requireWebGLRenderer().setRenderTarget(previousTarget);
    }

    if (target !== this.target) {
      // Swap back buffer and target
      [this.target, this.backTarget] = [this.backTarget, this.target];
    }
    return target;
  }

  // Read back the previously rendered target image as a Uint8Array of packed
  // RGBA values (in that order). Subsequent calls to this.readTarget()
  // will reuse the same buffers to minimize memory allocations.
  async readTarget(): Promise<Uint8Array> {
    if (!this.target) {
      throw new Error("Must initialize with target");
    }
    const { width, height } = this.target;
    const byteSize = width * height * 4;
    if (!this.superPixels || this.superPixels.length < byteSize) {
      this.superPixels = new Uint8Array(byteSize);
      // console.log(`Allocated superPixels: ${width}x${height} = ${pixelCount} bytes`);
    }
    const superPixels = this.superPixels;

    await this.requireWebGLRenderer().readRenderTargetPixelsAsync(
      this.target,
      0,
      0,
      width,
      height,
      superPixels,
    );

    const { superXY } = this;
    if (superXY === 1) {
      return superPixels;
    }

    const subWidth = width / superXY;
    const subHeight = height / superXY;
    const subSize = subWidth * subHeight * 4;
    if (!this.targetPixels || this.targetPixels.length < subSize) {
      this.targetPixels = new Uint8Array(subSize);
      // console.log(`Allocated targetPixels: ${subWidth}x${subHeight} = ${subSize} bytes`);
    }
    const targetPixels = this.targetPixels;

    const super2 = superXY * superXY;
    for (let y = 0; y < subHeight; y++) {
      const row = y * subWidth;
      for (let x = 0; x < subWidth; x++) {
        const superCol = x * superXY;
        let r = 0;
        let g = 0;
        let b = 0;
        let a = 0;
        for (let sy = 0; sy < superXY; sy++) {
          const superRow = (y * superXY + sy) * width;
          for (let sx = 0; sx < superXY; sx++) {
            const superIndex = (superRow + superCol + sx) * 4;
            r += superPixels[superIndex];
            g += superPixels[superIndex + 1];
            b += superPixels[superIndex + 2];
            a += superPixels[superIndex + 3];
          }
        }
        const pixelIndex = (row + x) * 4;
        targetPixels[pixelIndex] = r / super2;
        targetPixels[pixelIndex + 1] = g / super2;
        targetPixels[pixelIndex + 2] = b / super2;
        targetPixels[pixelIndex + 3] = a / super2;
      }
    }
    return targetPixels;
  }

  async renderReadTarget({
    scene,
    camera,
  }: {
    scene: THREE.Scene;
    camera: THREE.Camera;
  }): Promise<Uint8Array> {
    this.renderTarget({ scene, camera });
    return this.readTarget();
  }

  // Data and buffers used for environment map rendering
  private static cubeRender: {
    target: THREE.WebGLCubeRenderTarget;
    cubeCamera: THREE.CubeCamera;
    near: number;
    far: number;
  } | null = null;
  private static pmrem: THREE.PMREMGenerator | null = null;

  // Renders out the scene to a cube map that can be used for
  // Image-based lighting or similar applications. First optionally updates Gsplats,
  // sorts them with respect to the provided worldCenter, renders 6 cube faces.
  async renderCubeMap({
    scene,
    worldCenter,
    size = 256,
    near = 0.1,
    far = 1000,
    hideObjects = [],
    update = true,
    filter = false,
  }: {
    scene: THREE.Scene;
    worldCenter: THREE.Vector3;
    size?: number;
    near?: number;
    far?: number;
    hideObjects: THREE.Object3D[];
    update: boolean;
    filter: boolean;
  }): Promise<THREE.CubeTexture> {
    if (
      !SparkRenderer.cubeRender ||
      SparkRenderer.cubeRender.target.width !== size ||
      SparkRenderer.cubeRender.near !== near ||
      SparkRenderer.cubeRender.far !== far
    ) {
      if (SparkRenderer.cubeRender) {
        SparkRenderer.cubeRender.target.dispose();
      }
      const target = new THREE.WebGLCubeRenderTarget(size, {
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        generateMipmaps: filter,
        minFilter: filter ? THREE.LinearMipMapLinearFilter : THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        colorSpace: filter ? THREE.LinearSRGBColorSpace : THREE.SRGBColorSpace,
      });
      const cubeCamera = new THREE.CubeCamera(near, far, target);
      SparkRenderer.cubeRender = { target, cubeCamera, near, far };
    }

    const { target, cubeCamera } = SparkRenderer.cubeRender;
    cubeCamera.position.copy(worldCenter);

    // Save the visibility state of objects we want to hide before render
    const objectVisibility = new Map<THREE.Object3D, boolean>();
    for (const object of hideObjects) {
      objectVisibility.set(object, object.visible);
      object.visible = false;
    }

    if (update) {
      const tempCamera = new THREE.Camera();
      tempCamera.position.copy(worldCenter);
      await this.update({ scene, camera: tempCamera });
    }

    try {
      SparkRenderer.sparkOverride = this;
      // Update the CubeCamera, which performs 6 cube face renders
      cubeCamera.update(this.requireWebGLRenderer(), scene);
    } finally {
      SparkRenderer.sparkOverride = undefined;
    }

    // Restore viewpoint to default and object visibility
    for (const [object, visible] of objectVisibility.entries()) {
      object.visible = visible;
    }

    return target.texture;
  }

  async readCubeTargets(): Promise<Uint8Array[]> {
    if (!SparkRenderer.cubeRender) {
      throw new Error("No cube render");
    }

    const textures = SparkRenderer.cubeRender.target.texture;
    const promises = [];
    const buffers = [];

    for (let i = 0; i < textures.images.length; ++i) {
      const { width, height } = textures.images[i];
      const byteSize = width * height * 4;
      const readback = new Uint8Array(byteSize);
      buffers.push(readback);
      const promise = this.requireWebGLRenderer().readRenderTargetPixelsAsync(
        SparkRenderer.cubeRender.target,
        0,
        0,
        width,
        height,
        readback,
        i,
      );
      promises.push(promise);
    }

    await Promise.all(promises);
    return buffers;
  }

  // Renders out the scene to an environment map that can be used for
  // Image-based lighting or similar applications. First optionally updates Gsplats,
  // sorts them with respect to the provided worldCenter, renders 6 cube faces,
  // then pre-filters them using THREE.PMREMGenerator and returns a THREE.Texture
  // that can assigned directly to a THREE.MeshStandardMaterial.envMap property.
  async renderEnvMap({
    scene,
    worldCenter,
    size = 256,
    near = 0.1,
    far = 1000,
    hideObjects = [],
    update = true,
  }: {
    scene: THREE.Scene;
    worldCenter: THREE.Vector3;
    size?: number;
    near?: number;
    far?: number;
    hideObjects: THREE.Object3D[];
    update: boolean;
  }): Promise<THREE.Texture> {
    const cubeTexture = await this.renderCubeMap({
      scene,
      worldCenter,
      size,
      near,
      far,
      hideObjects,
      update,
      filter: true,
    });
    // Pre-filter the cube map using THREE.PMREMGenerator if requested
    if (!SparkRenderer.pmrem) {
      SparkRenderer.pmrem = new THREE.PMREMGenerator(
        this.requireWebGLRenderer(),
      );
    }

    return SparkRenderer.pmrem?.fromCubemap(cubeTexture).texture;
  }

  // Utility function to recursively set the envMap property for any
  // THREE.MeshStandardMaterial within the subtree of root.
  recurseSetEnvMap(root: THREE.Object3D, envMap: THREE.Texture) {
    root.traverse((node) => {
      if (node instanceof THREE.Mesh) {
        if (Array.isArray(node.material)) {
          for (const material of node.material) {
            if (material instanceof THREE.MeshStandardMaterial) {
              material.envMap = envMap;
            }
          }
        } else {
          if (node.material instanceof THREE.MeshStandardMaterial) {
            node.material.envMap = envMap;
          }
        }
      }
    });
  }

  async getLodTreeLevel(
    splats: SplatMesh,
    level: number,
    pageColoring = false,
  ) {
    const instance = this.lodInstances.get(splats);
    if (!instance) {
      return null;
    }

    const result = await this.ensureLodWorker().exclusive(async (worker) => {
      return await worker.call("getLodTreeLevel", {
        lodId: instance.lodId,
        level,
      });
    });

    if (splats.packedSplats?.lodSplats) {
      const newSplats = splats.packedSplats.lodSplats.extractSplats(
        result.indices,
        pageColoring,
      );
      return new SplatMesh({ packedSplats: newSplats });
    }
    if (splats.extSplats?.lodSplats) {
      const newSplats = splats.extSplats.lodSplats.extractSplats(
        result.indices,
        pageColoring,
      );
      return new SplatMesh({ extSplats: newSplats });
    }
    throw new Error(
      "Only LoD-enabled PackedSplats and ExtSplats are supported",
    );
  }

  /**
   * Raycast a SplatMesh against its LoD tree, down to the finest level that is
   * currently resident, instead of the coarse cut tested by SplatMesh.raycast().
   * Resolves to the closest intersection (all of them with closestOnly: false),
   * using a precision-safe ellipsoid test with semi-axes scaled by `sigma`
   * (default 1, the SplatMesh.raycast() criterion; at most 2).
   * With closestOnly: false the walk is capped at 65536 candidate nodes.
   * Falls back to SplatMesh.raycast() for meshes without a LoD tree here.
   * Requests are never dropped unless `coalesce` is set: then a newer request
   * with the same key (the string, or mesh.uuid for true) supersedes this one,
   * which rejects with an AbortError DOMException. Rejects likewise after
   * dispose().
   */
  async raycastAsync(
    mesh: SplatMesh,
    liveRaycaster: THREE.Raycaster,
    {
      closestOnly = true,
      sigma = 1,
      signal,
      coalesce,
    }: {
      closestOnly?: boolean;
      sigma?: number;
      signal?: AbortSignal;
      coalesce?: string | boolean;
    } = {},
  ): Promise<THREE.Intersection[]> {
    if (this.disposed) {
      throw new DOMException("SparkRenderer disposed", "AbortError");
    }
    // Callers reuse and move their raycaster while we await: snapshot it
    const raycaster = Object.assign(new THREE.Raycaster(), liveRaycaster, {
      ray: liveRaycaster.ray.clone(),
    });
    const key =
      coalesce === true
        ? mesh.uuid
        : typeof coalesce === "string"
          ? coalesce
          : undefined;
    const generation = ++this.pickCounter;
    if (key !== undefined) this.pickLatest.set(key, generation);
    const toIntersections = (distances: number[]) =>
      distances
        .sort((a, b) => a - b)
        .slice(0, closestOnly ? 1 : distances.length)
        .map((distance) => ({
          distance,
          point: raycaster.ray.at(distance, new THREE.Vector3()),
          object: mesh,
        }));

    const splats =
      mesh.packedSplats?.lodSplats ?? mesh.extSplats?.lodSplats ?? mesh.paged;
    const worldToMesh = mesh.matrixWorld.clone().invert();
    const origin = raycaster.ray.origin.clone().applyMatrix4(worldToMesh);
    const direction = raycaster.ray.direction
      .clone()
      .applyMatrix3(new THREE.Matrix3().setFromMatrix4(worldToMesh));
    // Covering radii in the LoD worker bound ellipsoids up to 2 sigma
    const raySigma = Math.min(sigma, 2);
    // One raycast buffer of candidates per call
    const maxCandidatesLimit = 65536;

    let far = raycaster.far;
    let closest = Number.POSITIVE_INFINITY;
    // ponytail: a retry restarts the walk (8x more candidates, far clamped to the
    // closest hit so far) instead of resuming it; resumable walk if this shows up.
    for (let maxCandidates = 64; ; ) {
      signal?.throwIfAborted();
      const result =
        splats &&
        mesh.raycastable &&
        wasm.isInitialized() &&
        this.lodIds.has(splats)
          ? await this.raycastPickWorker(
              splats,
              mesh,
              {
                origin: origin.toArray(),
                direction: direction.toArray(),
                near: raycaster.near,
                far,
                maxCandidates,
              },
              key,
              generation,
            ).then((result) =>
              !result
                ? result
                : {
                    hits: mesh.raycastLodNodes(
                      raycaster,
                      result.nodes,
                      raySigma,
                    ),
                    nextDistance: result.nextDistance,
                  },
            )
          : null;
      signal?.throwIfAborted();

      if (!result) {
        const intersects: Parameters<SplatMesh["raycast"]>[1] = [];
        mesh.raycast(raycaster, intersects);
        return toIntersections(intersects.map(({ distance }) => distance));
      }

      const { hits, nextDistance } = result;
      for (const hit of hits) {
        closest = Math.min(closest, hit);
      }
      if (
        nextDistance === Number.POSITIVE_INFINITY ||
        (closestOnly && closest <= nextDistance) ||
        maxCandidates === maxCandidatesLimit
      ) {
        return closestOnly
          ? toIntersections(Number.isFinite(closest) ? [closest] : [])
          : toIntersections(hits);
      }
      if (closestOnly) {
        far = Math.min(far, closest);
      }
      maxCandidates = Math.min(maxCandidates * 8, maxCandidatesLimit);
    }
  }

  /**
   * GPU pick prototype: renders a tiny (size x size) crop of the current
   * frame (same camera position/orientation, same splat ordering/LoD
   * selection already computed this frame) in two passes into two offscreen
   * float targets, then reconstructs the picked 3D point from the readback.
   * Resolves to null if the pick window had no coverage at all.
   *
   * Pass "media" composites splats back-to-front with standard premultiplied
   * "over" blending (the same blend the main splat shader uses), so R/A end
   * up holding a true visibility-weighted average depth (R = sum(w*d),
   * A = sum(w), w_i = a_i * product(1 - a_j) for every splat j in front of
   * i) - NOT a naive alpha-weighted sum, which would let splats behind an
   * opaque surface pull the average through/behind it.
   * Pass "solid" uses MIN blending to find the depth of the frontmost splat
   * with alpha > alphaThreshold, independent of the first pass. It runs only
   * with `solid: true` (default false = single "media" pass).
   *
   * x, y are CSS pixels relative to renderer.domElement (same convention as
   * mouse/pointer events), not canvas backing-store pixels.
   *
   * Prototype scope: only this SparkRenderer's splats are considered (every
   * other direct child of its parent scene is temporarily hidden for the
   * two synchronous render calls), and occlusion by non-splat opaque
   * geometry is not handled.
   */
  async pickAsync({
    camera,
    x,
    y,
    size = 5,
    alphaThreshold = 0.1,
    solid = false,
  }: {
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
    x: number;
    y: number;
    size?: number;
    alphaThreshold?: number;
    /** Also run the "solid" (first hit > alphaThreshold) pass. Default false:
     *  one pass only, pointSolida/distanceSolida are null. */
    solid?: boolean;
  }): Promise<SparkPickResult | null> {
    const t0 = performance.now();
    const canvas = this.renderer.domElement;
    const fullW = canvas.clientWidth || canvas.width;
    const fullH = canvas.clientHeight || canvas.height;

    const { pixelsMedia, pixelsSolid, floatTypeName, farBig } =
      await this.pickPixels({
        camera,
        fullW,
        fullH,
        ox: x - size / 2,
        oy: y - size / 2,
        w: size,
        h: size,
        alphaThreshold,
        solid,
      });

    const result = pickResultFromPixels({
      pixelsMedia,
      pixelsSolid,
      size,
      camera,
      x,
      y,
      fullW,
      fullH,
      farBig,
      floatType: floatTypeName,
    });
    if (result) {
      result.ms = performance.now() - t0;
    }
    return result;
  }

  /**
   * Batched variant of pickAsync() for several screen points at once: ONE
   * "media" render + ONE readback of the window covering every point's
   * size x size pick window, then per point exactly the pixel selection and
   * depth -> world reconstruction pickAsync() does on its own window.
   * x, y are CSS pixels relative to renderer.domElement, as in pickAsync().
   * Results are in input order; null = no coverage, or the point's pick
   * window is not entirely inside the canvas. No "solid" pass
   * (pointSolida/distanceSolida are always null). `ms` is the whole call
   * time, the same for every result.
   * Pixel grid: every window is aligned to the first in-canvas point's
   * sub-pixel offset; points at integer offsets from it (the usual case,
   * e.g. integer CSS coordinates) see exactly the pixels pickAsync() would,
   * others are rounded to the nearest pixel.
   */
  async pickPointsAsync({
    camera,
    points,
    size = 5,
    alphaThreshold = 0.1,
  }: {
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
    points: { x: number; y: number }[];
    size?: number;
    alphaThreshold?: number;
  }): Promise<(SparkPickResult | null)[]> {
    const t0 = performance.now();
    const canvas = this.renderer.domElement;
    const fullW = canvas.clientWidth || canvas.width;
    const fullH = canvas.clientHeight || canvas.height;
    const half = size / 2;

    // Integer offset (in pixels) of each point's window from the first valid
    // window's top-left corner; null = window not entirely in the canvas.
    let baseX = 0;
    let baseY = 0;
    let hasBase = false;
    let minCol = Number.POSITIVE_INFINITY;
    let minRow = Number.POSITIVE_INFINITY;
    let maxCol = Number.NEGATIVE_INFINITY;
    let maxRow = Number.NEGATIVE_INFINITY;
    const cells = points.map(({ x, y }) => {
      if (
        x - half < 0 ||
        y - half < 0 ||
        x + half > fullW ||
        y + half > fullH
      ) {
        return null;
      }
      if (!hasBase) {
        baseX = x - half;
        baseY = y - half;
        hasBase = true;
      }
      const col = Math.round(x - half - baseX);
      const row = Math.round(y - half - baseY);
      minCol = Math.min(minCol, col);
      minRow = Math.min(minRow, row);
      maxCol = Math.max(maxCol, col);
      maxRow = Math.max(maxRow, row);
      return { col, row };
    });
    if (!hasBase) return points.map(() => null);

    const w = maxCol - minCol + size;
    const h = maxRow - minRow + size;
    const {
      pixelsMedia: pixels,
      floatTypeName,
      farBig,
    } = await this.pickPixels({
      camera,
      fullW,
      fullH,
      ox: baseX + minCol,
      oy: baseY + minRow,
      w,
      h,
      alphaThreshold,
      solid: false,
    });

    const results = points.map(({ x, y }, index) => {
      const cell = cells[index];
      if (!cell) return null;
      // Copy this point's size x size window out of the shared readback.
      // Rows come back bottom-to-top: window row r (counted from its bottom)
      // is shared row h - size - top + r, top = rows from the shared top.
      const left = cell.col - minCol;
      const top = cell.row - minRow;
      const pixelsMedia = new Float32Array(size * size * 4);
      for (let r = 0; r < size; r++) {
        const src = ((h - size - top + r) * w + left) * 4;
        pixelsMedia.set(pixels.subarray(src, src + size * 4), r * size * 4);
      }
      return pickResultFromPixels({
        pixelsMedia,
        pixelsSolid: null,
        size,
        camera,
        x,
        y,
        fullW,
        fullH,
        farBig,
        floatType: floatTypeName,
      });
    });
    const ms = performance.now() - t0;
    for (const result of results) {
      if (result) result.ms = ms;
    }
    return results;
  }

  // Renders and reads back the pick window on either backend. Pixels are
  // Float32 RGBA, rows bottom to top (WebGL readRenderTargetPixels layout).
  private async pickPixels(
    args: Parameters<SparkRenderer["pickRender"]>[0],
  ): Promise<{
    pixelsMedia: Float32Array;
    pixelsSolid: Float32Array | null;
    floatTypeName: "float" | "half";
    farBig: number;
  }> {
    const { camera, w, h, solid } = args;
    const farBig = pickFarBig(camera);
    if (this.native) {
      const {
        media,
        solid: pixelsSolid,
        floatType,
      } = await this.native.pickPixels(camera, { ...args, farBig });
      return {
        pixelsMedia: media,
        pixelsSolid,
        floatTypeName: floatType,
        farBig,
      };
    }
    const renderer = this.requireWebGLRenderer();
    const { targetMedia, targetSolid, floatOk, floatTypeName } =
      this.pickRender(args);
    // Safe to await now: render targets are read directly regardless of
    // the renderer's currently-bound render target.
    const [pixelsMedia, pixelsSolid] = await Promise.all([
      readPickTargetPixels(renderer, targetMedia, w, h, floatOk),
      solid ? readPickTargetPixels(renderer, targetSolid, w, h, floatOk) : null,
    ]);
    return { pixelsMedia, pixelsSolid, floatTypeName, farBig };
  }

  // Synchronous part shared by pickAsync()/pickPointsAsync(): renders the
  // w x h CSS-pixel crop of the full frame whose top-left corner is
  // (ox, oy) into the pick target(s). Nothing here awaits, so the
  // renderer's state swap is never observable from outside.
  private pickRender({
    camera,
    fullW,
    fullH,
    ox,
    oy,
    w,
    h,
    alphaThreshold,
    solid,
  }: {
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
    fullW: number;
    fullH: number;
    ox: number;
    oy: number;
    w: number;
    h: number;
    alphaThreshold: number;
    solid: boolean;
  }) {
    const renderer = this.requireWebGLRenderer();

    // Lazily create the two pick materials. Both share the exact same
    // uniforms *objects* as the main material (spread of `this.uniforms`,
    // not a deep clone), so whatever onBeforeRender() writes into e.g.
    // `this.uniforms.near.value` every frame is immediately visible here
    // too, on top of their own alphaThreshold/pickSolidMode.
    if (!this.pickMaterialMedia) {
      const shaders = getShaders();
      const commonParams = {
        glslVersion: THREE.GLSL3,
        vertexShader: shaders.splatVertex,
        fragmentShader: shaders.splatPickFragment,
        transparent: true,
        depthTest: this.material.depthTest,
        depthWrite: this.material.depthWrite,
        side: THREE.DoubleSide,
        allowOverride: false,
        // Per-fragment max-density depth along the pixel ray instead of the
        // flat splat-center depth (see splatVertex.glsl/splatPickFragment.glsl).
        defines: {
          ...(this.material as THREE.ShaderMaterial).defines,
          SPARK_PICK_DEPTH: "",
        },
      };
      // Weighted-average pass: standard premultiplied "over" blending, the
      // same (ONE, ONE_MINUS_SRC_ALPHA) THREE uses for NormalBlending with
      // premultipliedAlpha true - no CustomBlending needed.
      this.pickMaterialMedia = new THREE.ShaderMaterial({
        ...commonParams,
        premultipliedAlpha: true,
        uniforms: {
          ...this.uniforms,
          alphaThreshold: { value: alphaThreshold },
          pickSolidMode: { value: false },
        },
      });
      // First-solid-hit pass: MIN blend keeps the smallest depth written.
      // WebGL ignores src/dst factors for Min/Max equations; left at ONE.
      this.pickMaterialSolid = new THREE.ShaderMaterial({
        ...commonParams,
        uniforms: {
          ...this.uniforms,
          alphaThreshold: { value: alphaThreshold },
          pickSolidMode: { value: true },
        },
        blending: THREE.CustomBlending,
        blendEquation: THREE.MinEquation,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneFactor,
        blendEquationAlpha: THREE.MinEquation,
        blendSrcAlpha: THREE.OneFactor,
        blendDstAlpha: THREE.OneFactor,
      });
    }
    // Local consts so TS keeps the non-undefined narrowing from the lazy
    // init above across the rest of this method.
    const pickMaterialMedia = this.pickMaterialMedia;
    const pickMaterialSolid = this.pickMaterialSolid;
    if (!pickMaterialMedia || !pickMaterialSolid) {
      throw new Error("pick materials not initialized");
    }
    (pickMaterialMedia.uniforms.alphaThreshold.value as number) =
      alphaThreshold;
    (pickMaterialSolid.uniforms.alphaThreshold.value as number) =
      alphaThreshold;

    // EXT_float_blend is required to blend into a FloatType target on many
    // platforms (plain float rendering without blending doesn't need it);
    // fall back to HalfFloatType otherwise.
    const floatOk = renderer.extensions.has("EXT_float_blend");
    const pickFloatType = floatOk ? THREE.FloatType : THREE.HalfFloatType;
    const floatTypeName: "float" | "half" = floatOk ? "float" : "half";

    const makeTarget = () =>
      new THREE.WebGLRenderTarget(w, h, {
        type: pickFloatType,
        format: THREE.RGBAFormat,
        depthBuffer: true,
        stencilBuffer: false,
        magFilter: THREE.NearestFilter,
        minFilter: THREE.NearestFilter,
      });

    let targetMedia = this.pickTargetMedia;
    let targetSolid = this.pickTargetSolid;
    if (
      !targetMedia ||
      !targetSolid ||
      this.pickTargetW !== w ||
      this.pickTargetH !== h ||
      this.pickTargetIsFloat !== floatOk
    ) {
      targetMedia?.dispose();
      targetSolid?.dispose();
      targetMedia = makeTarget();
      targetSolid = makeTarget();
      this.pickTargetMedia = targetMedia;
      this.pickTargetSolid = targetSolid;
      this.pickTargetW = w;
      this.pickTargetH = h;
      this.pickTargetIsFloat = floatOk;
    }

    const pickCamera = camera.clone();
    pickCamera.setViewOffset(fullW, fullH, ox, oy, w, h);
    pickCamera.updateProjectionMatrix();

    const farBig = pickFarBig(camera);

    // Hide every other direct child of the scene for these two render
    // calls: this SparkRenderer already batches every splat found in the
    // scene into a single instanced draw, so rendering just `this` is
    // enough to get all splats into the pick targets without other scene
    // geometry (which the pick shaders/blend modes don't know how to
    // handle) polluting them.
    const scene = this.parent;
    const hidden: THREE.Object3D[] = [];
    if (scene) {
      for (const child of scene.children) {
        if (child !== (this as THREE.Object3D) && child.visible) {
          hidden.push(child);
          child.visible = false;
        }
      }
    }

    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    const prevXrEnabled = renderer.xr.enabled;
    const prevClearColor = new THREE.Color();
    renderer.getClearColor(prevClearColor);
    const prevClearAlpha = renderer.getClearAlpha();
    const prevMaterial = this.material as THREE.ShaderMaterial;
    // `material` is declared readonly in the public API; this is an
    // internal, same-tick swap for the pick render only.
    const materialHolder = this as unknown as {
      material: THREE.ShaderMaterial;
    };

    // Everything from here to the matching restore below runs
    // synchronously (no `await` in between), so the renderer's "current"
    // state is never observable as changed from outside this function - in
    // particular, the main render loop can't run concurrently on a
    // partially-swapped state. Both passes share one try/finally.
    const prevClipXY = this.clipXY;
    try {
      renderer.xr.enabled = false;
      this.pickPass = true;
      // Disable center-based frustum clipping for picking: a splat whose
      // quad covers the cursor can still have its center outside the tiny
      // setViewOffset crop, and the vertex shader's clipXY cull would drop
      // it before SPARK_PICK_DEPTH's exact quad-vs-NDC test ever runs.
      this.clipXY = 1e6;
      renderer.autoClear = false;

      // Pass "media": clear to (0,0,0,0), premultiplied-over blend.
      materialHolder.material = pickMaterialMedia;
      renderer.setRenderTarget(targetMedia);
      renderer.setClearColor(0x000000, 0);
      renderer.clear(true, true, false);
      renderer.render(scene ?? this, pickCamera);

      if (solid) {
        // Pass "solid" (only with solid: true): clear R(GBA) to farBig via the renderer's low-level
        // state API - renderer.setClearColor()/Color only accept [0,1]
        // components, which can't express a large sentinel, and this also
        // sidesteps any doubt about alpha-specific clear clamping, since the
        // sentinel is read back from the *color* (R) channel here.
        // premultipliedAlpha=false so the (r,g,b)*=a scaling THREE applies
        // for a premultiplied *canvas context* doesn't zero it out (our
        // clear alpha is 0).
        materialHolder.material = pickMaterialSolid;
        renderer.setRenderTarget(targetSolid);
        renderer.state.buffers.color.setClear(farBig, farBig, farBig, 0, false);
        renderer.clear(true, true, false);
        renderer.render(scene ?? this, pickCamera);
      }
    } finally {
      this.pickPass = false;
      this.clipXY = prevClipXY;
      materialHolder.material = prevMaterial;
      renderer.setRenderTarget(prevTarget);
      renderer.autoClear = prevAutoClear;
      renderer.xr.enabled = prevXrEnabled;
      // Goes through the normal (cached) path, so it also corrects the
      // state cache after the low-level setClear() call above.
      renderer.setClearColor(prevClearColor, prevClearAlpha);
      for (const child of hidden) child.visible = true;
    }

    return { targetMedia, targetSolid, floatOk, floatTypeName, farBig };
  }

  /**
   * One raycastLodTree on the pick worker, at most one in flight, the others
   * waiting in FIFO order. A request with a coalesce `key` replaces the waiting
   * request with the same key (latest wins), so a burst of hovers never piles
   * up, and requests of a superseded raycastAsync() `generation` stop at their
   * next round trip: both reject with an AbortError, as do waiting requests on
   * dispose(). Requests without a key are never dropped.
   * No lock needed for the tree state: record (rootPage) is
   * read and the message posted in the same tick, and every tree update or
   * disposal is mirrored to the pick worker in the same tick it changes lodIds,
   * so its FIFO always holds the tree state this record describes.
   */
  private async raycastPickWorker(
    splats: PackedSplats | ExtSplats | PagedSplats,
    mesh: SplatMesh,
    ray: Omit<
      Parameters<rpcHandlers["raycastLodTree"]>[0],
      "lodId" | "rootPage"
    >,
    key: string | undefined,
    generation: number,
  ) {
    const superseded = () => new DOMException("Pick superseded", "AbortError");
    if (this.pickBusy) {
      const index =
        key === undefined
          ? -1
          : this.pickQueue.findIndex((queued) => queued.key === key);
      if (index >= 0) this.pickQueue.splice(index, 1)[0].go(false);
      const go = await new Promise<boolean>((resolve) => {
        this.pickQueue.push({ key, go: resolve });
      });
      // On go the finishing request handed its slot over: pickBusy stayed true
      if (!go) throw superseded();
    }
    this.pickBusy = true;
    try {
      if (this.disposed) {
        throw new DOMException("SparkRenderer disposed", "AbortError");
      }
      if (key !== undefined && this.pickLatest.get(key) !== generation) {
        throw superseded();
      }
      this.ensureLodWorker();
      const record = this.lodIds.get(splats);
      if (
        !this.pickWorker ||
        !record ||
        (mesh.paged && record.rootPage === undefined)
      ) {
        return null;
      }
      return await this.pickWorker.call("raycastLodTree", {
        ...ray,
        lodId: record.lodId,
        rootPage: record.rootPage,
      });
    } catch (error) {
      // dispose() terminated the worker under the in-flight request
      throw this.disposed
        ? new DOMException("SparkRenderer disposed", "AbortError")
        : error;
    } finally {
      const next = this.pickQueue.shift();
      if (next) next.go(true);
      else this.pickBusy = false;
    }
  }

  get premultipliedAlpha(): boolean {
    return this.material.premultipliedAlpha;
  }

  set premultipliedAlpha(value: boolean) {
    if (this.material.premultipliedAlpha !== value) {
      this.material.premultipliedAlpha = value;
      this.material.needsUpdate = true;
    }
  }
}

// Three.js sets this property on the ordinary THREE.WebGLRenderTarget it
// creates for an XR session, so instanceof cannot identify it.
function checkIsXRRenderTarget(renderTarget: THREE.RenderTarget | null) {
  return (renderTarget as unknown as Record<string, boolean>)?.isXRRenderTarget;
}

// Reconstructs a view-space ray for pixel (ndcX, ndcY) in NDC space ([-1,1])
// from the camera's own (unmodified, non-viewOffset) projection matrix, so
// it matches whatever the camera was actually showing at full resolution.
// Returns a point such that moving `d` world units along it reaches
// view-space z = -d (i.e. walks the ray to the plane z_view = -d), matching
// the linear depth splatPickFragment.glsl outputs.
function viewPointAtDepth(
  camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
  ndcX: number,
  ndcY: number,
  d: number,
): THREE.Vector3 {
  const m = camera.projectionMatrix.elements;
  if ("isOrthographicCamera" in camera) {
    // Ortho projection: clipX = m00*viewX + m03; clipY = m11*viewY + m13
    // (clipW === 1, so NDC === clip). Rays are parallel: (viewX, viewY, -d).
    const viewX = (ndcX - m[12]) / m[0];
    const viewY = (ndcY - m[13]) / m[5];
    return new THREE.Vector3(viewX, viewY, -d);
  }
  // Perspective projection: clipX = m00*viewX + m02*viewZ, clipW = -viewZ.
  // At viewZ = -1: ndcX = m00*viewX - m02 => viewX = (ndcX + m02) / m00.
  const viewX = (ndcX + m[8]) / m[0];
  const viewY = (ndcY + m[9]) / m[5];
  return new THREE.Vector3(viewX * d, viewY * d, -d);
}

// Keep this in sync with splatPickFragment.glsl's farBig(): both the clear
// value and the shader's "no solid hit" sentinel must agree so readback can
// tell a real depth from "nothing there". The WGSL pass takes it as a uniform.
function pickFarBig(camera: THREE.Camera & { far: number }) {
  return Math.min(60000, Math.max(1e4, camera.far * 4));
}

// Reads back a pickAsync() render target as a plain Float32Array, regardless
// of whether it's FloatType (read directly) or HalfFloatType (read as raw
// 16-bit words and expanded one by one - there's no native typed array for
// half floats).
async function readPickTargetPixels(
  renderer: THREE.WebGLRenderer,
  target: THREE.WebGLRenderTarget,
  w: number,
  h: number,
  floatOk: boolean,
): Promise<Float32Array> {
  const numComponents = w * h * 4;
  if (floatOk) {
    const pixels = new Float32Array(numComponents);
    await renderer.readRenderTargetPixelsAsync(target, 0, 0, w, h, pixels);
    return pixels;
  }
  const halfPixels = new Uint16Array(numComponents);
  await renderer.readRenderTargetPixelsAsync(target, 0, 0, w, h, halfPixels);
  const pixels = new Float32Array(numComponents);
  for (let i = 0; i < numComponents; i++) {
    pixels[i] = THREE.DataUtils.fromHalfFloat(halfPixels[i]);
  }
  return pixels;
}

function pickResultFromPixels({
  pixelsMedia,
  pixelsSolid,
  size,
  camera,
  x,
  y,
  fullW,
  fullH,
  farBig,
  floatType,
}: {
  // Layout per pixel (premultiplied-over pass): R = sum(w*d), A = sum(w).
  pixelsMedia: Float32Array;
  // Layout per pixel (MIN-blended pass): R = depth of first solid hit, or
  // farBig() if none. null when the solid pass was skipped (solid: false).
  pixelsSolid: Float32Array | null;
  size: number;
  camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
  x: number;
  y: number;
  fullW: number;
  fullH: number;
  farBig: number;
  floatType: "float" | "half";
}): SparkPickResult | null {
  const center = Math.floor(size / 2);
  const solidThreshold = farBig * 0.5;

  // Coverage (the "media" pass's accumulated alpha, channel A) decides
  // whether a pixel counts as a hit at all. Prefer the center pixel (the
  // actual click/hover point); row/col order (WebGL returns rows
  // bottom-to-top) doesn't matter here since the center index is symmetric
  // for odd `size`, the default.
  let bestRow = -1;
  let bestCol = -1;
  const centerCoverage = pixelsMedia[(center * size + center) * 4 + 3];
  if (centerCoverage > 1e-4) {
    bestRow = center;
    bestCol = center;
  } else {
    // Otherwise the pixel with the most coverage, closest to center on ties.
    let bestCoverage = 0;
    let bestDistSq = Number.POSITIVE_INFINITY;
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const coverage = pixelsMedia[(row * size + col) * 4 + 3];
        if (coverage <= 1e-4) continue;
        const distSq = (row - center) ** 2 + (col - center) ** 2;
        if (
          coverage > bestCoverage ||
          (coverage === bestCoverage && distSq < bestDistSq)
        ) {
          bestCoverage = coverage;
          bestDistSq = distSq;
          bestRow = row;
          bestCol = col;
        }
      }
    }
  }

  if (bestRow < 0) {
    return null;
  }

  const i = (bestRow * size + bestCol) * 4;
  const wSum = pixelsMedia[i]; // sum(w*d)
  const coverage = pixelsMedia[i + 3]; // sum(w)
  // depth of first solid hit, or farBig(); null if the solid pass was skipped
  const solidDepth = pixelsSolid ? pixelsSolid[i] : null;

  const dMedia = coverage > 1e-4 ? wSum / coverage : null;
  const dSolida =
    solidDepth !== null && solidDepth < solidThreshold ? solidDepth : null;

  // ndcX/ndcY for the *center* of the pick window, in the full-resolution
  // image's NDC space - the camera's real, unmodified projection matrix is
  // used for this, not the cropped pickCamera's.
  const ndcX = (x / fullW) * 2 - 1;
  const ndcY = -(y / fullH) * 2 + 1;

  const pointMedia =
    dMedia !== null
      ? viewPointAtDepth(camera, ndcX, ndcY, dMedia).applyMatrix4(
          camera.matrixWorld,
        )
      : new THREE.Vector3();
  const pointSolida =
    dSolida !== null
      ? viewPointAtDepth(camera, ndcX, ndcY, dSolida).applyMatrix4(
          camera.matrixWorld,
        )
      : null;

  return {
    point: pointMedia,
    pointMedia,
    pointSolida,
    distanceMedia: dMedia ?? Number.POSITIVE_INFINITY,
    distanceSolida: dSolida,
    copertura: coverage,
    floatType,
    ms: 0,
  };
}
