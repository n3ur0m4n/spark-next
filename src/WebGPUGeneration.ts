import * as THREE from "three";
import { SplatPager } from "./SplatPager";
import { calcDispatchSize } from "./WebGPUUtils";
import { splatDefinesWgsl } from "./shaders-wgsl";
import generateSourceBuffers from "./shaders/generateSourceBuffers.wgsl?raw";
import generateSourcePaged from "./shaders/generateSourcePaged.wgsl?raw";
import generateSplats from "./shaders/generateSplats.wgsl?raw";

/** Immutable CPU description of one ordinary mesh in an accumulator generation. */
export type NativeMeshGeneration = {
  source: object;
  revision: number;
  primary: Uint32Array;
  secondary?: Uint32Array;
  sh: (Uint32Array | undefined)[];
  sourceCount: number;
  sourceExt: boolean;
  base: number;
  count: number;
  indices?: Uint32Array;
  /** Changes whenever the contents of indices change (same array reused). */
  indicesVersion?: number;
  rotation: number[];
  translationScale: number[];
  localViewOrigin: number[];
  encoding: number[];
  shMaxOpacity: number[];
  numSh: number;
  recolor: number[];
  /** SplatMesh.introDots time in seconds, negative when off. */
  introDots: number;
  /** Paged source: read the pager's page textures; primary/secondary/sh are unused. */
  pager?: SplatPager;
  /** SplatEdits.editData (numEdits uvec4) followed by its sdfData (8 uvec4 per SDF). */
  edits?: { data: Uint32Array; numEdits: number; numSdfs: number };
};

type PagerTextures = {
  textures: (THREE.DataArrayTexture | undefined)[];
  gpu: (GPUTexture | undefined)[];
  views: GPUTextureView[];
};

type SourceBuffers = {
  revision: number;
  arrays: (Uint32Array | undefined)[];
  sourceCount: number;
  primary: GPUBuffer;
  secondary: GPUBuffer;
  sh: GPUBuffer;
  shOffsets: number[];
};

/** Source uploads are shared across instances; output ownership stays with the backend slot. */
export class WebGPUGeneration {
  private readonly sources = new Map<object, SourceBuffers>();
  private readonly pipeline: GPUComputePipeline;
  private pagedPipeline: GPUComputePipeline | null = null;
  private readonly pagers = new Map<SplatPager, PagerTextures>();
  private dummyPages: GPUTexture | null = null;
  private readonly dummy: GPUBuffer;
  private readonly meshes: {
    uniform: GPUBuffer;
    indices: GPUBuffer;
    /** What indices holds, to skip re-uploading an unchanged selection. */
    indicesKey?: [Uint32Array, number | undefined, number];
    edits: GPUBuffer;
  }[] = [];
  private disposed = false;

  constructor(private readonly device: GPUDevice) {
    this.dummy = this.buffer(16, GPUBufferUsage.STORAGE, "native-empty-source");
    this.pipeline = device.createComputePipeline({
      label: "spark-native-generate",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: `${splatDefinesWgsl}\n${generateSourceBuffers}\n${generateSplats}`,
        }),
        entryPoint: "generate",
      },
    });
  }

  private getPagedPipeline() {
    this.pagedPipeline ??= this.device.createComputePipeline({
      label: "spark-native-generate-paged",
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({
          code: `${splatDefinesWgsl}\n${generateSourcePaged}\n${generateSplats}`,
        }),
        entryPoint: "generate",
      },
    });
    return this.pagedPipeline;
  }

  private destroyPager(state: PagerTextures) {
    for (const texture of state.gpu) texture?.destroy();
  }

  /**
   * Mirror SplatPager's page textures. The pager fills CPU layers and marks
   * them with addLayerUpdate() exactly as for WebGL; only those layers are
   * copied here, then the marks are cleared.
   */
  private syncPager(pager: SplatPager): GPUTextureView[] {
    const maxLayers = this.device.limits.maxTextureArrayLayers;
    if (pager.maxPages > maxLayers)
      throw new Error(
        `Spark: ${pager.maxPages} splat pages exceed maxTextureArrayLayers ${maxLayers}; lower maxPagedSplats or use THREE.WebGLRenderer.`,
      );
    let state = this.pagers.get(pager);
    if (!state) {
      state = { textures: [], gpu: [], views: [] };
      this.pagers.set(pager, state);
    }
    this.dummyPages ??= this.device.createTexture({
      label: "spark-native-empty-pages",
      size: [1, 1, 1],
      format: "rgba32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    const dummy = this.dummyPages;
    // Packed pagers have three SH slots, extended ones four.
    const slots = [pager.packedTexture, pager.extTexture, ...pager.shTextures];
    for (let i = 0; i < 6; i++) {
      const texture = slots[i]?.value;
      if (
        !texture ||
        texture === SplatPager.emptyUint32x4 ||
        texture === SplatPager.emptyUint32x2
      ) {
        state.gpu[i]?.destroy();
        state.gpu[i] = undefined;
        state.textures[i] = undefined;
        state.views[i] = dummy.createView({ dimension: "2d-array" });
        continue;
      }
      const words = texture.format === THREE.RGIntegerFormat ? 2 : 4;
      const { width, height } = texture.image;
      let gpu = state.gpu[i];
      const layers = new Set(texture.layerUpdates);
      if (!gpu || state.textures[i] !== texture) {
        // A new GPU mirror (first use, or after release while the scene had
        // no paged mesh) needs every resident page, not only dirty ones.
        pager.pageToSplatsChunk.forEach((resident, page) => {
          if (resident) layers.add(page);
        });
        gpu?.destroy();
        gpu = this.device.createTexture({
          label: "spark-native-pages",
          size: [width, height, pager.maxPages],
          format: words === 2 ? "rg32uint" : "rgba32uint",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        state.gpu[i] = gpu;
        state.textures[i] = texture;
        state.views[i] = gpu.createView({ dimension: "2d-array" });
      }
      const layerWords = width * height * words;
      const data = texture.image.data as Uint32Array;
      for (const layer of layers) {
        this.device.queue.writeTexture(
          { texture: gpu, origin: [0, 0, layer] },
          data.subarray(layer * layerWords, (layer + 1) * layerWords),
          { bytesPerRow: width * words * 4, rowsPerImage: height },
          [width, height, 1],
        );
      }
      texture.layerUpdates.clear();
    }
    return state.views;
  }

  private buffer(size: number, usage: GPUBufferUsageFlags, label: string) {
    const bytes = Math.max(16, Math.ceil(size / 4) * 4);
    if (
      bytes > this.device.limits.maxBufferSize ||
      ((usage & GPUBufferUsage.STORAGE) !== 0 &&
        bytes > this.device.limits.maxStorageBufferBindingSize)
    ) {
      throw new Error(
        `Spark: ${label} needs ${bytes} bytes, exceeding native device limits; use THREE.WebGLRenderer.`,
      );
    }
    return this.device.createBuffer({
      size: bytes,
      usage,
      label: `spark-${label}`,
    });
  }

  private destroySource(buffers: SourceBuffers) {
    buffers.primary.destroy();
    if (buffers.secondary !== this.dummy) buffers.secondary.destroy();
    if (buffers.sh !== this.dummy) buffers.sh.destroy();
  }

  private uploadSource(mesh: NativeMeshGeneration): SourceBuffers {
    const arrays = [mesh.primary, mesh.secondary, ...mesh.sh];
    const old = this.sources.get(mesh.source);
    if (
      old &&
      old.revision === mesh.revision &&
      old.sourceCount === mesh.sourceCount &&
      arrays.every((array, i) => old.arrays[i] === array)
    )
      return old;
    const created: GPUBuffer[] = [];
    const upload = (
      array: Uint32Array | undefined,
      size = array?.byteLength ?? 0,
    ) => {
      if (!array || size === 0) return this.dummy;
      const buffer = this.buffer(
        size,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "native-source",
      );
      created.push(buffer);
      this.device.queue.writeBuffer(
        buffer,
        0,
        array.buffer,
        array.byteOffset,
        size,
      );
      return buffer;
    };
    try {
      const primary = upload(mesh.primary, mesh.sourceCount * 16);
      const secondary = upload(mesh.secondary, mesh.sourceCount * 16);
      const shOffsets: number[] = [];
      let shWords = 0;
      for (const band of mesh.sh) {
        shOffsets.push(shWords);
        shWords += band?.length ?? 0;
      }
      let sh = this.dummy;
      if (shWords) {
        sh = this.buffer(
          shWords * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          "native-source-sh",
        );
        created.push(sh);
        mesh.sh.forEach((band, i) => {
          if (band?.length)
            this.device.queue.writeBuffer(
              sh,
              shOffsets[i] * 4,
              band.buffer,
              band.byteOffset,
              band.byteLength,
            );
        });
      }
      const result = {
        revision: mesh.revision,
        arrays,
        sourceCount: mesh.sourceCount,
        primary,
        secondary,
        sh,
        shOffsets,
      };
      this.sources.set(mesh.source, result);
      if (old) this.destroySource(old);
      return result;
    } catch (error) {
      for (const buffer of created) buffer.destroy();
      throw error;
    }
  }

  encode(
    encoder: GPUCommandEncoder,
    args: {
      meshes: NativeMeshGeneration[];
      output: GPUTextureView;
      output2: GPUTextureView;
      depths: GPUBuffer;
      outputExt: boolean;
      viewOrigin: number[];
      viewDirection: number[];
      sortRadial: boolean;
      sortCullCos: number;
      timestampWrites?: () => GPUComputePassTimestampWrites | undefined;
    },
  ) {
    if (this.disposed) throw new Error("Spark: native generation is disposed");
    const live = new Set(args.meshes.map((mesh) => mesh.source));
    for (const [source, buffers] of this.sources) {
      if (!live.has(source)) {
        this.destroySource(buffers);
        this.sources.delete(source);
      }
    }
    const livePagers = new Set(args.meshes.map((mesh) => mesh.pager));
    for (const [pager, state] of this.pagers) {
      if (!livePagers.has(pager)) {
        this.destroyPager(state);
        this.pagers.delete(pager);
      }
    }
    while (this.meshes.length > args.meshes.length) {
      const state = this.meshes.pop();
      if (state) {
        state.uniform.destroy();
        if (state.indices !== this.dummy) state.indices.destroy();
        if (state.edits !== this.dummy) state.edits.destroy();
      }
    }
    for (let i = 0; i < args.meshes.length; i++) {
      const mesh = args.meshes[i];
      if (!mesh.count) continue;
      const paged = mesh.pager ? this.syncPager(mesh.pager) : null;
      const source = paged ? null : this.uploadSource(mesh);
      let state = this.meshes[i];
      if (!state) {
        state = {
          uniform: this.buffer(
            208,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            "native-mesh-uniform",
          ),
          indices: this.dummy,
          edits: this.dummy,
        };
        this.meshes[i] = state;
      }
      if (mesh.indices) {
        const bytes = Math.max(16, mesh.count * 4);
        if (state.indices === this.dummy || state.indices.size < bytes) {
          if (state.indices !== this.dummy) state.indices.destroy();
          state.indices = this.buffer(
            bytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "native-selection",
          );
          state.indicesKey = undefined;
        }
        const key = state.indicesKey;
        if (
          // Unversioned indices may be rewritten in place: always upload.
          mesh.indicesVersion === undefined ||
          !key ||
          key[0] !== mesh.indices ||
          key[1] !== mesh.indicesVersion ||
          key[2] < mesh.count
        ) {
          this.device.queue.writeBuffer(
            state.indices,
            0,
            mesh.indices.buffer,
            mesh.indices.byteOffset,
            mesh.count * 4,
          );
          state.indicesKey = [mesh.indices, mesh.indicesVersion, mesh.count];
        }
      }
      const edits = mesh.edits?.numEdits ? mesh.edits : undefined;
      if (edits) {
        const bytes = edits.data.byteLength;
        if (state.edits === this.dummy || state.edits.size < bytes) {
          if (state.edits !== this.dummy) state.edits.destroy();
          state.edits = this.buffer(
            bytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "native-edits",
          );
        }
        this.device.queue.writeBuffer(
          state.edits,
          0,
          edits.data.buffer,
          edits.data.byteOffset,
          bytes,
        );
      }
      const data = new ArrayBuffer(208);
      const u32 = new Uint32Array(data);
      const f32 = new Float32Array(data);
      u32.set([
        mesh.base,
        mesh.count,
        mesh.sourceCount,
        mesh.indices ? 1 : 0,
        mesh.sourceExt ? 1 : 0,
        args.outputExt ? 1 : 0,
        args.sortRadial ? 1 : 0,
        mesh.numSh,
      ]);
      f32.set(mesh.rotation, 8);
      f32.set(mesh.translationScale, 12);
      f32.set(args.viewOrigin, 16);
      f32.set(args.viewDirection, 20);
      f32[23] = args.sortCullCos;
      f32.set(mesh.localViewOrigin, 24);
      f32.set(mesh.encoding, 28);
      f32.set(mesh.shMaxOpacity, 32);
      if (source) u32.set(source.shOffsets, 36);
      f32.set(mesh.recolor, 40);
      if (edits) u32.set([edits.numEdits, edits.numSdfs], 44);
      f32[48] = mesh.introDots;
      this.device.queue.writeBuffer(state.uniform, 0, data);
      const pipeline = paged ? this.getPagedPipeline() : this.pipeline;
      // Sparse binding numbers: paged textures occupy 8-10 only when paged.
      const resources: [number, GPUBuffer | GPUTextureView][] = [
        [0, state.uniform],
        ...(source
          ? [source.primary, source.secondary, source.sh]
          : (paged as GPUTextureView[]).slice(0, 3)
        ).map((resource, i): [number, GPUBuffer | GPUTextureView] => [
          i + 1,
          resource,
        ]),
        [4, mesh.indices ? state.indices : this.dummy],
        [5, args.output],
        [6, args.output2],
        [7, args.depths],
        ...(paged ? paged.slice(3, 6) : []).map(
          (view, i): [number, GPUTextureView] => [i + 8, view],
        ),
        [11, edits ? state.edits : this.dummy],
      ];
      const group = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: resources.map(([binding, resource]) => ({
          binding,
          resource:
            resource instanceof GPUBuffer ? { buffer: resource } : resource,
        })),
      });
      const pass = encoder.beginComputePass({
        label: "spark-native-generate",
        timestampWrites: args.timestampWrites?.(),
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      const [x, y] = calcDispatchSize(
        Math.ceil(mesh.count / 256),
        this.device.limits.maxComputeWorkgroupsPerDimension,
      );
      pass.dispatchWorkgroups(x, y);
      pass.end();
    }
  }

  /** Release source ownership when the committed scene becomes empty. */
  clearSources() {
    for (const source of this.sources.values()) this.destroySource(source);
    this.sources.clear();
    for (const state of this.pagers.values()) this.destroyPager(state);
    this.pagers.clear();
    for (const mesh of this.meshes) {
      mesh.uniform.destroy();
      if (mesh.indices !== this.dummy) mesh.indices.destroy();
    }
    this.meshes.length = 0;
  }

  dispose() {
    this.disposed = true;
    this.clearSources();
    this.dummy.destroy();
    this.dummyPages?.destroy();
  }
}
