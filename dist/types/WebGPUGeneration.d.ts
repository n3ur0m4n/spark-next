import { SplatPager } from "./SplatPager";
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
    edits?: {
        data: Uint32Array;
        numEdits: number;
        numSdfs: number;
    };
};
/** Source uploads are shared across instances; output ownership stays with the backend slot. */
export declare class WebGPUGeneration {
    private readonly device;
    private readonly sources;
    private readonly pipeline;
    private pagedPipeline;
    private readonly pagers;
    private dummyPages;
    private readonly dummy;
    private readonly meshes;
    private disposed;
    constructor(device: GPUDevice);
    private getPagedPipeline;
    private destroyPager;
    /**
     * Mirror SplatPager's page textures. The pager fills CPU layers and marks
     * them with addLayerUpdate() exactly as for WebGL; only those layers are
     * copied here, then the marks are cleared.
     */
    private syncPager;
    private buffer;
    private destroySource;
    private uploadSource;
    encode(encoder: GPUCommandEncoder, args: {
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
    }): void;
    /** Release source ownership when the committed scene becomes empty. */
    clearSources(): void;
    dispose(): void;
}
