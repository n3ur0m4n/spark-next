// Archivio .ssog = streamed SOG (lod-meta.json + cartelle {lod}_{i}/) impacchettato in un
// unico ZIP "stored" (come .3tz). Le entry si leggono con richieste HTTP Range: stessa logica
// di createZipIndex/readZipEntry di SINTESI (src/utils/tiles3tz.js).

const SIG_EOCD = 0x06054b50;
const SIG_ZIP64_LOCATOR = 0x07064b50;
const SIG_ZIP64_EOCD = 0x06064b50;
const SIG_CENTRAL = 0x02014b50;
const SIG_LOCAL = 0x04034b50;
const TAIL_BYTES = 66000; // EOCD + commento massimo (64 KB)
const LOCAL_HEADER_MARGIN = 30 + 512; // header locale fisso + nome/extra stimati

/**
 * Legge byte dall'archivio. `start < 0` = ultimi `-start` byte (Range a suffisso).
 * `end` esclusivo. Ritorna anche la dimensione totale del file.
 */
export type SsogRead = (
  start: number,
  end?: number,
) => Promise<{ bytes: Uint8Array; total: number }>;

export type SsogEntry = { offset: number; size: number };

export function isSsogUrl(url: string): boolean {
  return /\.ssog$/i.test(url.split(/[?#]/)[0]);
}

function findBackwards(dv: DataView, sig: number): number {
  for (let i = dv.byteLength - 4; i >= 0; i--) {
    if (dv.getUint32(i, true) === sig) return i;
  }
  return -1;
}

const view = (b: Uint8Array) =>
  new DataView(b.buffer, b.byteOffset, b.byteLength);

/** Indice della central directory: path interno → offset header locale + dimensione */
export async function readSsogIndex(
  read: SsogRead,
): Promise<Map<string, SsogEntry>> {
  const { bytes: tail, total } = await read(-TAIL_BYTES);
  const tdv = view(tail);
  const eocd = findBackwards(tdv, SIG_EOCD);
  if (eocd < 0) throw new Error("ssog: EOCD non trovato, non è uno ZIP");

  let entries = tdv.getUint16(eocd + 10, true);
  let cdSize = tdv.getUint32(eocd + 12, true);
  let cdOffset = tdv.getUint32(eocd + 16, true);
  if (entries === 0xffff || cdOffset === 0xffffffff) {
    const loc = findBackwards(tdv, SIG_ZIP64_LOCATOR);
    if (loc < 0) throw new Error("ssog: locator ZIP64 non trovato");
    const z64Off = Number(tdv.getBigUint64(loc + 8, true));
    const zdv = view((await read(z64Off, z64Off + 56)).bytes);
    if (zdv.getUint32(0, true) !== SIG_ZIP64_EOCD) {
      throw new Error("ssog: firma EOCD64 non valida");
    }
    entries = Number(zdv.getBigUint64(32, true));
    cdSize = Number(zdv.getBigUint64(40, true));
    cdOffset = Number(zdv.getBigUint64(48, true));
  }
  if (cdOffset + cdSize > total)
    throw new Error("ssog: central directory fuori file");

  // Central directory intera in una sola richiesta
  const cd = (await read(cdOffset, cdOffset + cdSize)).bytes;
  const dv = view(cd);
  const decoder = new TextDecoder();
  const index = new Map<string, SsogEntry>();
  let p = 0;
  for (let i = 0; i < entries; i++) {
    if (dv.getUint32(p, true) !== SIG_CENTRAL) {
      throw new Error(`ssog: firma central directory non valida (entry ${i})`);
    }
    if (dv.getUint16(p + 10, true) !== 0) {
      throw new Error("ssog: entry compressa, serve ZIP stored");
    }
    let size = dv.getUint32(p + 20, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    let offset = dv.getUint32(p + 42, true);
    const nameStart = p + 46;
    const name = decoder.decode(cd.subarray(nameStart, nameStart + nameLen));

    if (size === 0xffffffff || offset === 0xffffffff) {
      // Extra ZIP64 (id 0x0001): solo i campi saturi, ordine uncompressed, compressed, offset
      const uncompSat = dv.getUint32(p + 24, true) === 0xffffffff;
      let ep = nameStart + nameLen;
      const extraEnd = ep + extraLen;
      while (ep < extraEnd) {
        const id = dv.getUint16(ep, true);
        const len = dv.getUint16(ep + 2, true);
        if (id === 0x0001) {
          let dp = ep + 4;
          if (uncompSat) dp += 8;
          if (size === 0xffffffff) {
            size = Number(dv.getBigUint64(dp, true));
            dp += 8;
          }
          if (offset === 0xffffffff) offset = Number(dv.getBigUint64(dp, true));
          break;
        }
        ep += 4 + len;
      }
    }
    index.set(name.replace(/\\/g, "/").replace(/^\/+/, ""), { offset, size });
    p = nameStart + nameLen + extraLen + commentLen;
  }
  return index;
}

/** Byte di una entry stored: una sola Range con margine per l'header locale */
export async function readSsogEntry(
  read: SsogRead,
  entry: SsogEntry,
): Promise<Uint8Array> {
  const { bytes } = await read(
    entry.offset,
    entry.offset + LOCAL_HEADER_MARGIN + entry.size,
  );
  const dv = view(bytes);
  if (dv.getUint32(0, true) !== SIG_LOCAL) {
    throw new Error("ssog: firma header locale non valida");
  }
  const skip = 30 + dv.getUint16(26, true) + dv.getUint16(28, true);
  if (skip + entry.size > bytes.length) {
    throw new Error("ssog: entry troncata");
  }
  return bytes.slice(skip, skip + entry.size);
}

/** SsogRead su HTTP: pretende 206, altrimenti il server ha ignorato il Range */
export function httpSsogRead(
  url: string,
  init: { headers?: Record<string, string>; credentials: RequestCredentials },
  signal?: AbortSignal,
): SsogRead {
  return async (start, end) => {
    const headers = new Headers(init.headers);
    headers.set(
      "Range",
      start < 0 ? `bytes=${start}` : `bytes=${start}-${(end as number) - 1}`,
    );
    const res = await fetch(url, {
      headers,
      credentials: init.credentials,
      signal,
      cache: "no-store",
    });
    if (res.status !== 206) {
      throw new Error(
        `ssog: Range non onorato (status ${res.status}) su "${url}"`,
      );
    }
    // "bytes a-b/TOTALE"
    const total = Number(res.headers.get("content-range")?.split("/")[1]);
    return { bytes: new Uint8Array(await res.arrayBuffer()), total };
  };
}
