import type * as THREE from "three";
export interface LodMetaData {
    /** URL assoluti di filenames[i] (meta.json della cartella SOG) */
    fileUrls: string[];
    numLeaves: number;
    numLevels: number;
    /** min xyz, max xyz per foglia (spazio PLY del file) */
    bounds: Float32Array;
    /** per foglia×asse: eccesso della semi-estensione sulla mediana (PC nodeBoundsExcess) */
    excess: Float32Array;
    /** per foglia×livello: file (indice in fileUrls, -1 = livello senza dati), offset, count */
    file: Int32Array;
    offset: Uint32Array;
    count: Uint32Array;
    /** livello LoD dei dati di ogni file */
    fileLevel: Int32Array;
    /** livello più fine ammesso: parte dal più grossolano (bootstrap), poi 0 */
    rangeMin: number;
    /** livello mostrato per foglia (-1 = niente), stato come PC nodeInfo.currentLod */
    current: Int16Array;
    /** tabella banda → livello per rangeMin (PC GSplatLodTable) */
    table?: {
        rangeMin: number;
        bandLod: Int16Array;
        bandCount: Uint32Array;
    };
}
type LodMetaNode = {
    bound: {
        min: number[];
        max: number[];
    };
    children?: LodMetaNode[];
    lods?: Record<string, {
        file: number;
        offset: number;
        count: number;
    }>;
};
export declare function isLodMetaUrl(url: string): boolean;
/** Compensazione FOV PC: min(tanHalfV, tanHalfH) / tan(22.5°) */
export declare function lodMetaFovScale(tanHalfVFov: number, aspect: number): number;
export declare function parseLodMeta(json: {
    lodLevels: number;
    filenames: string[];
    tree: LodMetaNode;
}, baseUrl: string): LodMetaData;
/**
 * Distanza mondo camera → foglia come PC evaluateNodeDistances: punto più vicino
 * dell'AABB ristretto (lodDistanceShrink), penalità dietro camera, × scala mondo
 * del mesh × compensazione FOV.
 */
export declare function lodMetaDistances(data: LodMetaData, viewToObject: THREE.Matrix4, fovScale?: number): Float64Array;
export interface LodMetaInstance<S> {
    splats: S;
    data: LodMetaData;
    /** camera → spazio locale della mesh */
    viewToObject: THREE.Matrix4;
    lodScale: number;
}
export interface LodMetaResult<S> {
    /** indici splat da disegnare per istanza (page*65536 + offset + k) */
    numSplats: number[];
    indices: Uint32Array[];
    /** file da tenere/caricare, già ordinati per priorità */
    priority: {
        splats: S;
        chunk: number;
    }[];
    /** prefetch del passo successivo verso l'ottimo (priorità più bassa) */
    prefetch: {
        splats: S;
        chunk: number;
    }[];
    /** bootstrap finito: range allargato, va rifatta la selezione */
    rangeChanged: boolean;
}
/**
 * Selezione stile PlayCanvas: banda = floor(1 + log_3(d / (s·b))) clampata al
 * range, livello dalla tabella (buchi/livello vuoto), s globale = la più grande
 * che sta nel budget splat (e pagine). Underfill fino a 5 passi più grossolani
 * già residenti; se il voluto non è pronto resta il livello mostrato prima.
 * `residentPage(splats, file)` = pagina del file già caricata, o -1.
 */
export declare function selectLodMeta<S>({ instances, maxSplats, maxFiles, residentPage, indicesCache, fovScale, lodLimit, }: {
    instances: LodMetaInstance<S>[];
    maxSplats: number;
    maxFiles: number;
    residentPage: (splats: S, file: number) => number;
    /** buffer indici riusabili per istanza (stesso ordine), può essere più corto */
    indicesCache?: (Uint32Array | undefined)[];
    /** compensazione FOV, vedi lodMetaFovScale */
    fovScale?: number;
    /** limita il dettaglio massimo a c=0 (come radix budgetMode "limit"): il budget può solo abbassarlo, mai superarlo */
    lodLimit?: boolean;
}): LodMetaResult<S>;
export {};
