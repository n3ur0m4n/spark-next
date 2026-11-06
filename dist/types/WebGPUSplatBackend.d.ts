import { type NativeMeshGeneration } from "./WebGPUGeneration";
type UniformBuffers = {
    spark: GPUBuffer;
    proj: GPUBuffer;
    frag: GPUBuffer;
};
/** Raw pickAsync() readback, laid out like WebGL's (rows bottom to top). */
export type WebGPUPickPixels = {
    media: Float32Array;
    solid: Float32Array | null;
    floatType: "float" | "half";
};
export type WebGPUOrderingTarget = "active" | "working" | "write";
/**
 * WebGPU rendering backend for SparkRenderer.
 *
 * Owns all GPU-specific state: pipeline, textures, buffers, and draw calls.
 * Generation evaluates source SH and LoD before this draw stage.
 */
export declare class WebGPUSplatBackend {
    private static readonly MAX_TIMESTAMP_PASSES;
    device: GPUDevice;
    pipeline: GPURenderPipeline | null;
    bindGroupLayout: GPUBindGroupLayout | null;
    bindGroup: GPUBindGroup | null;
    private bindGroupLayoutVersion;
    pipelineFormat: GPUTextureFormat | null;
    private pipelineHalvedAlphaState;
    private pipelinePremultipliedAlphaState;
    private renderSlots;
    private activeSlotIndex;
    private workingSlotIndex;
    private pendingRenderSlot;
    private nextGeneration;
    webgpuDebug: boolean;
    splatTexture: GPUTexture | null;
    splatTextureView: GPUTextureView | null;
    splatTexSize: {
        width: number;
        height: number;
        depth: number;
    } | null;
    splatTexture2: GPUTexture | null;
    splatTextureView2: GPUTextureView | null;
    splatTex2Size: {
        width: number;
        height: number;
        depth: number;
    } | null;
    orderingTexture: GPUTexture | null;
    orderingRows: number;
    private syncDepthReadbackBuffer;
    private readonly countReadbacks;
    halvedAlpha: boolean;
    /**
     * Mirrors SparkRenderer's public `premultipliedAlpha` option. Drives both the
     * fragment `PREMULTIPLIED_ALPHA` shader constant and the blend descriptor, so WebGPU
     * composites the same way WebGL does when an app opts out.
     */
    premultipliedAlpha: boolean;
    /** Sort-free stochastic draw (SparkRenderer stochasticActive) for the next frame. */
    stochastic: boolean;
    /** Same shaders, opaque output with depth writes (WebGL: NoBlending + depthWrite). */
    private stochasticPipeline;
    splatEncoding: [number, number, number, number];
    sparkUniformBuffer: GPUBuffer;
    projMatrixBuffer: GPUBuffer;
    fragUniformBuffer: GPUBuffer;
    vertexBuffer: GPUBuffer;
    indexBuffer: GPUBuffer;
    depthTexture: GPUTexture | null;
    /** Non-sRGB copy of an sRGB color target, for renderSeparatePass(srgbBlend). */
    private srgbScratch;
    depthTextureView: GPUTextureView | null;
    /** Externally-owned scene depth (three.js), adopted via setExternalDepthTexture(). */
    private externalDepthTexture;
    private externalDepthView;
    /** Count of frames where the splat draw was skipped, keyed by reason. */
    readonly drawSkipped: Record<string, number>;
    /** Reason recorded for the most recent skipped draw. */
    lastDrawSkippedReason: string | null;
    /** True once the WebGPU device has been reported lost. */
    deviceLost: boolean;
    /** Message from the device-lost event, if any. */
    deviceLostReason: string | null;
    /** Count of uncaptured WebGPU validation/out-of-memory errors observed. */
    uncapturedErrorCount: number;
    /** Message of the first uncaptured error, retained for the debug snapshot. */
    firstUncapturedError: string | null;
    /** True after dispose(); guards in-flight async completions. */
    private disposed;
    private readonly releaseDeviceLost;
    /** Why the last offered external depth texture was rejected, if it was. */
    externalDepthRejectReason: string | null;
    depthWidth: number;
    depthHeight: number;
    private gpuSorter;
    /** 16-bit OneSweep sort; null where unsupported (multipass 32-bit). */
    private oneSweep;
    /** "multipass" forces the portable 32-bit sort (A/B and pixel tests). */
    gpuSortMode: "auto" | "multipass";
    /** Key width of the GPU radix sort in use: 16 (OneSweep) or 32. */
    get gpuSortKeyBits(): 16 | 32;
    private nativeRangeBuffer;
    private nativeCountBuffer;
    private nativeSortArgs;
    private nativeBlockCounts;
    private nativeCompactValues;
    private bindGroupCache;
    private resourceIds;
    private nextResourceId;
    /** Latches pipeline failures; capacity checks still apply to each generation. */
    private gpuSortUnavailable;
    private gpuSortValidated;
    private gpuSortPreparation;
    private packOrderingPipeline;
    private packOrderingLayout;
    private packOrderingUniform;
    /** The draw mode belongs to the same committed slot as its ordering/count. */
    get useIndirectDraw(): boolean;
    set useIndirectDraw(value: boolean);
    private generatedOutputDummy;
    private nativeGeneration;
    private nativeFillPipeline;
    private nativeKeysPipeline;
    private nativeCountPipeline;
    private nativeCompactPipelines;
    private nativeSortUniform;
    private sparkUniformData;
    private fragUniformData;
    private readonly onUncapturedError;
    constructor(device: GPUDevice);
    private activeRenderSlot;
    private writeRenderSlot;
    private depthRenderSlot;
    private orderingRenderSlot;
    private syncLegacyFields;
    private markRenderSlotsDirty;
    private debugLog;
    beginRenderSlotUpdate(reason?: string): void;
    commitRenderSlot(): void;
    cancelRenderSlotUpdate(expectedGeneration?: number): boolean;
    hasPendingRenderSlot(): boolean;
    hasSplatData(): boolean;
    hasDepthSplatData(): boolean;
    getDepthSlotGeneration(): number;
    getActiveSlotGeneration(): number;
    getWorkingSlotGeneration(): number;
    /**
     * Back-to-front alpha blending for the splat draw.
     *
     * Mirrors what three.js programs for `NormalBlending`: premultiplied source
     * colors blend with `ONE`, straight-alpha ones with `SRC_ALPHA`. The alpha
     * channel accumulates identically either way.
     */
    private splatBlendState;
    /**
     * Create the render pipeline on first use (needs canvas format).
     */
    ensurePipeline(colorFormat: GPUTextureFormat): void;
    /**
     * Write SparkUniforms + projection matrix + fragment uniforms to GPU.
     */
    updateUniforms(uniforms: {
        renderSize: {
            value: {
                x: number;
                y: number;
            };
        };
        renderToViewQuat: {
            value: {
                x: number;
                y: number;
                z: number;
                w: number;
            };
        };
        renderToViewPos: {
            value: {
                x: number;
                y: number;
                z: number;
            };
        };
        maxStdDev: {
            value: number;
        };
        renderToViewBasis: {
            value: {
                elements: number[];
            };
        };
        minPixelRadius: {
            value: number;
        };
        maxPixelRadius: {
            value: number;
        };
        enableExtSplats: {
            value: boolean;
        };
        enableCovSplats: {
            value: boolean;
        };
        time: {
            value: number;
        };
        deltaTime: {
            value: number;
        };
        debugFlag: {
            value: boolean;
        };
        minAlpha: {
            value: number;
        };
        enable2DGS: {
            value: boolean;
        };
        lodInflate: {
            value: boolean;
        };
        blurAmount: {
            value: number;
        };
        preBlurAmount: {
            value: number;
        };
        focalDistance: {
            value: number;
        };
        apertureAngle: {
            value: number;
        };
        clipXY: {
            value: number;
        };
        focalAdjustment: {
            value: number;
        };
        encodeLinear: {
            value: boolean;
        };
        near: {
            value: number;
        };
        far: {
            value: number;
        };
        falloff: {
            value: number;
        };
    }, projectionMatrix: {
        elements: number[];
    }, isOrthographic: boolean, target?: UniformBuffers): void;
    /**
     * Upload the ordering texture from sort worker results.
     */
    uploadOrdering(data: Uint32Array, width: number, rows: number, target?: WebGPUOrderingTarget): void;
    /**
     * Ensure we have a depth texture of the right size.
     */
    private ensureDepthTexture;
    /**
     * Device-loss-safe buffer mapping. `mapAsync()` rejects when the device is
     * lost or the buffer is destroyed mid-map; unguarded `await`s therefore turn
     * a lost device into unhandled rejections and leaked staging buffers.
     *
     * Returns true if the buffer is mapped and safe to read.
     */
    mapBufferAsync(buffer: GPUBuffer, mode?: GPUMapModeFlags): Promise<boolean>;
    /**
     * Split a workgroup count across X and Y so it never exceeds
     * `maxComputeWorkgroupsPerDimension` (65535 by default).
     *
     * A single-dimension dispatch ceilings at 65535 * 256 = 16,776,960 splats,
     * which is reachable: LoD source sets of ~20M splats already exist. Shaders
     * reconstruct the linear index as `(wid.y * nwg.x + wid.x) * 256 + lid.x`, so
     * this must stay in sync with splatDepth.wgsl / splatGather.wgsl.
     *
     * Uses a near-square split so the padding never exceeds one row of groups.
     */
    computeDispatchSize(workgroups: number): [number, number];
    /** Enable GPU timestamp timing. No effect without the device feature. */
    gpuTimingEnabled: boolean;
    private timestampQuerySet;
    private timestampResolveBuffer;
    private timestampReadbackBuffer;
    private timestampSlotLabels;
    private timestampSlotsUsed;
    private timestampReadInFlight;
    /** Last completed per-pass GPU durations in ms, keyed by pass label. */
    lastGpuPassMs: Record<string, number>;
    /** Span of timed Spark passes; excludes untimed host work and final conversion. */
    lastGpuFrameMs: number | null;
    gpuTimingSequence: number;
    gpuTimingGeneration: number | null;
    /** True when timestamp timing can actually be recorded on this device. */
    get gpuTimingAvailable(): boolean;
    private ensureTimestampResources;
    /**
     * Reserve a timestamp pair for a pass. Returns the descriptor fragment to
     * spread into a pass descriptor, or undefined when timing is off/exhausted.
     */
    private timestampWritesFor;
    /**
     * Resolve the frame's timestamps. Call once per frame after the last timed
     * pass has been submitted. Reads back asynchronously; frames that arrive
     * while a read is in flight are skipped rather than queued.
     */
    resolveGpuTiming(): void;
    /** True when three.js scene depth is currently bound for occlusion. */
    hasExternalDepth(): boolean;
    /**
     * Device limits and features relevant to splat rendering, for debug captures.
     *
     * Actual limits and feature negotiation are part of capture provenance;
     * source and generated storage are checked against these limits before use.
     */
    describeLimits(): Record<string, unknown>;
    /** Record a skipped splat draw so captures can distinguish it from flicker. */
    private noteDrawSkipped;
    /**
     * Adopt an externally-owned depth buffer (three.js's scene depth) as this
     * pass's depth attachment, so splats are occluded by scene geometry.
     *
     * This replaces copying: `depth24plus` is not a copyable format in WebGPU, so
     * `copyTextureToTexture` on it is a validation error. Binding three's depth
     * texture directly is both correct and cheaper. Our pipeline declares
     * `depthWriteEnabled: false`, so three's depth contents are never modified.
     *
     * Pass `null` to fall back to the backend's own cleared depth texture.
     * Returns true if the texture was accepted.
     */
    setExternalDepthTexture(texture: GPUTexture | null, targetWidth?: number, targetHeight?: number): boolean;
    /**
     * Rebuild the bind group when textures change.
     */
    private rebuildBindGroup;
    /** Generate ordinary meshes using the same output layout/encoding as WebGL. */
    generateNative(args: {
        meshes: NativeMeshGeneration[];
        numSplats: number;
        outputExt: boolean;
        viewOrigin: number[];
        viewDirection: number[];
        sortRadial: boolean;
        /** cos(sortCullFov / 2), or -2 when cone culling is off. */
        sortCullCos: number;
    }): void;
    /**
     * Test hook: a COPY_SRC buffer holding the generated splat texels in
     * linear splat order (the caller destroys it).
     */
    copyGeneratedOutput(second?: boolean): GPUBuffer;
    readNativeDepths(count: number): Promise<Uint32Array<ArrayBuffer>>;
    /** Explicit diagnostic readback; never inserted into the normal radix loop. */
    readNativeDrawCount(): Promise<number>;
    private ensureNativeSortPipelines;
    /**
     * Bind groups keyed by pipeline and resource identity. Resources only
     * change on reallocation, so steady-state sorts create no bind groups.
     */
    private cachedBindGroup;
    private resourceId;
    private prepareNativeKeys;
    /**
     * OneSweep input: compact the valid (uncull) splats in index order into
     * 16-bit keys quantized between the view's min/max depth, plus their
     * indices. The GPU writes the visible count, the sort/pack dispatch args
     * and the slot's draw args.
     */
    private compactNativeKeys;
    private createDrawArgs;
    /**
     * True from a radix sort's submit until the GPU has executed it. Like the
     * WebGL worker sort, the next sort of an unchanged mapping waits for it.
     */
    sortInFlight: boolean;
    /**
     * Give the working slot the committed radix ordering and draw count: a new
     * generation of the same splat mapping draws with the previous order, as
     * WebGL displays a regenerated accumulator before its sort completes.
     */
    reuseActiveOrdering(orderingRows: number): boolean;
    private countNativeDraw;
    /**
     * Execute a separate render pass that composites splats onto the
     * provided color texture, preserving existing content (loadOp: "load").
     *
     * srgbBlend (sRGB color texture only): blend on the stored sRGB-encoded
     * bytes, like WebGL drawing into an sRGB canvas, instead of in linear space.
     * The target is copied to a non-sRGB texture of the same bytes, the splats
     * (sRGB-encoded colors, pipeline in the non-sRGB format) blend there, and the
     * bytes are copied back.
     */
    renderSeparatePass(colorTexture: GPUTexture, numInstances: number, preloadedDepth?: boolean, srgbBlend?: boolean): void;
    private pickUniforms;
    private pickLayout;
    private pickPipelines;
    private pickTargets;
    /** Separate uniform buffers so a pick never disturbs the frame's uniforms. */
    pickUniformBuffers(): UniformBuffers;
    private uniformBuffer;
    private pickPipeline;
    /**
     * Draws the active slot (same ordering and draw count as the last frame)
     * with the pick shaders into width x height float targets and reads them
     * back. Uniforms must already be in pickUniformBuffers(). No depth test:
     * like WebGL's pick pass, only splats are considered.
     */
    readPickPixels({ width, height, numInstances, alphaThreshold, farBig, solid, }: {
        width: number;
        height: number;
        numInstances: number;
        alphaThreshold: number;
        farBig: number;
        solid: boolean;
    }): Promise<WebGPUPickPixels>;
    /**
     * Fail early, and actionably, when a buffer would exceed device limits.
     *
     * Each generated output and each source binding must fit the host device.
     * Reducing the selected LoD budget does not reduce source storage.
     */
    private assertStorageBufferFits;
    /**
     * Ensure the splat texture exists at the right size.
     * Allocates generation output before copyBufferToTexture.
     */
    ensureSplatTexture(numSplats: number): {
        width: number;
        height: number;
        depth: number;
    };
    /**
     * Ensure the ext splat texture exists at the right size.
     */
    ensureSplatTexture2(numSplats: number): {
        width: number;
        height: number;
        depth: number;
    };
    ensureGpuSort(): Promise<boolean>;
    private prepareGpuSort;
    private ensurePackOrderingPipeline;
    private orderingViews;
    private orderingStorageView;
    private ensureGpuOrderingTexture;
    /**
     * Depth keys + portable GPU radix + pack into the ordering texture.
     * No mapAsync. Returns false if the GPU sort path cannot run.
     */
    private nativeKeysBuffer;
    private sortSequence;
    gpuDepthSortAndPack(params: {
        numSplats: number;
        viewOrigin: {
            x: number;
            y: number;
            z: number;
        };
        viewDir: {
            x: number;
            y: number;
            z: number;
        };
        sortRadial: boolean;
        enableExtSplats: boolean;
        orderingRows: number;
        orderingTarget?: WebGPUOrderingTarget;
    }): Promise<boolean>;
    dispose(): void;
}
export {};
