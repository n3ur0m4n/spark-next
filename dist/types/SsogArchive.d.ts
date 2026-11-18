/**
 * Legge byte dall'archivio. `start < 0` = ultimi `-start` byte (Range a suffisso).
 * `end` esclusivo. Ritorna anche la dimensione totale del file.
 */
export type SsogRead = (start: number, end?: number) => Promise<{
    bytes: Uint8Array;
    total: number;
}>;
export type SsogEntry = {
    offset: number;
    size: number;
};
export declare function isSsogUrl(url: string): boolean;
/** Indice della central directory: path interno → offset header locale + dimensione */
export declare function readSsogIndex(read: SsogRead): Promise<Map<string, SsogEntry>>;
/** Byte di una entry stored: una sola Range con margine per l'header locale */
export declare function readSsogEntry(read: SsogRead, entry: SsogEntry): Promise<Uint8Array>;
/** SsogRead su HTTP: pretende 206, altrimenti il server ha ignorato il Range */
export declare function httpSsogRead(url: string, init: {
    headers?: Record<string, string>;
    credentials: RequestCredentials;
}, signal?: AbortSignal): SsogRead;
