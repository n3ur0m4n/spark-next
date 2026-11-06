import * as THREE from "three";

import { decode_rad_header } from "spark-rs";
import { type LodMetaData, isLodMetaUrl, parseLodMeta } from "./LodMeta";
import {
  FileBytesPagedSplatsFetcher,
  type PagedSplatsFetcher,
  UrlPagedSplatsFetcher,
} from "./PagedSplatsFetcher";
import { getSplatFileType, getSplatFileTypeFromPath } from "./SplatLoader";
import type { SplatSource } from "./SplatMesh";
import type { SplatPager } from "./SplatPager";
import { workerPool } from "./SplatWorker";
import {
  type SsogEntry,
  type SsogRead,
  httpSsogRead,
  isSsogUrl,
  readSsogEntry,
  readSsogIndex,
} from "./SsogArchive";
import {
  DEFAULT_SPLAT_ENCODING,
  LN_SCALE_MAX,
  LN_SCALE_MIN,
  type RadMeta,
  type SplatEncoding,
  SplatFileType,
} from "./defines";
import * as dyno from "./dyno";
import { decodeExtSplat, unpackSplat, uploadU32DataTextureRows } from "./utils";
import * as wasm from "./wasm";

// Diagnostica [lod-diag] per il caricamento dei chunk .ssog paged (vedi fetchDecodeChunk sotto);
// cap a 2000 righe totali per non intasare il log su caricamenti molto lunghi.
let lodDiagLineCount = 0;
function lodDiagLine(
  label: string,
  chunk: number,
  data?: Record<string, unknown>,
) {
  if (lodDiagLineCount++ >= 2000) return;
  console.warn("[lod-diag]", label, "chunk", chunk, data ?? "");
}

/**
 * Options for constructing {@link PagedSplats}. At least one source needs to be provided.
 * When multiple sources are given, only one takes effect. Order of precedence:
 *  * {@link PagedSplatsOptions#fileBytes `fileBytes`}
 *  * {@link PagedSplatsOptions#fetchSource `fetchSource`}
 *  * {@link PagedSplatsOptions#rootUrl `rootUrl`}
 */
export interface PagedSplatsOptions {
  pager?: SplatPager;
  /**
   * The root URL to fetch the paged splats from. Chunks are expected
   * to be relative to this URL.
   */
  rootUrl?: string;
  /**
   * Additional headers to set when making fetch requests.
   * @note Only relevant when using a {@link PagedSplatsOptions#rootUrl `rootUrl`}.
   */
  requestHeader?: Record<string, string>;
  /**
   * Whether credentials should be sent when making fetch requests.
   * @note Only relevant when using a {@link PagedSplatsOptions#rootUrl `rootUrl`}.
   */
  withCredentials?: boolean;
  /**
   * Raw bytes of the paged splats file.
   */
  fileBytes?: Uint8Array;
  /**
   * Custom paged splats fetcher to use.
   */
  fetchSource?: PagedSplatsFetcher;
  /**
   * The file type of the paged splats. This is required when using a {@link PagedSplatsOptions#fetchSource `fetchSource`}.
   * In case of {@link PagedSplatsOptions#fileBytes `fileBytes`} or a {@link PagedSplatsOptions#rootUrl `rootUrl`} the file type
   * is optional and will be automatically inferred unless provided.
   */
  fileType?: SplatFileType;
  /**
   * Maximum Spherical Harmonics level to use.
   * @default 3
   */
  maxSh?: number;
  /** Formato LoD PlayCanvas "streamed SOG" (default: url che finisce con lod-meta.json o .ssog) */
  lodMeta?: boolean;
}

export class PagedSplats implements SplatSource {
  pager?: SplatPager;
  readonly fileType: SplatFileType;
  readonly fetchSource: PagedSplatsFetcher;

  numSh: number;
  maxSh: number;
  sh1Codes?: Uint32Array;
  sh2Codes?: Uint32Array;
  sh3Codes?: Uint32Array | [Uint32Array, Uint32Array];

  numSplats: number;
  /** Bumped by update(): the native path re-uploads indices only on change. */
  indicesVersion = 0;
  splatEncoding?: SplatEncoding;
  radMetaPromise?: Promise<{ meta: RadMeta; chunksStart: number }>;
  // lod-meta.json: albero foglie appiattito (undefined finché non caricato),
  // 1 chunk = 1 file filenames[i]. Selezione in LodMeta.selectLodMeta.
  lodMeta?: LodMetaData;
  lodMetaPromise?: Promise<LodMetaData>;
  lodMetaIndices?: Uint32Array;
  lodMetaError?: unknown;
  // .ssog: archivio ZIP unico, entry lette a Range; prefix = url archivio + "/"
  ssog?: { read: SsogRead; index: Map<string, SsogEntry>; prefix: string };

  readonly dynoNumSplats: dyno.DynoInt<"numSplats">;
  readonly dynoIndices: dyno.DynoUsampler2D<"indices", THREE.DataTexture>;
  readonly rgbMinMaxLnScaleMinMax: dyno.DynoVec4<
    THREE.Vector4,
    "rgbMinMaxLnScaleMinMax"
  >;
  readonly lodOpacity: dyno.DynoBool<"lodOpacity">;
  readonly dynoNumSh: dyno.DynoInt<"numSh">;
  readonly shMax: dyno.DynoVec3<THREE.Vector3, "shMax">;

  readonly abortController: AbortController = new AbortController();

  constructor(options: PagedSplatsOptions) {
    this.pager = options.pager;
    this.numSh = 0;
    this.maxSh = options.maxSh ?? 3;

    this.numSplats = 0;

    this.dynoNumSplats = new dyno.DynoInt({ value: 0 });
    this.dynoIndices = new dyno.DynoUsampler2D({
      value: PagedSplats.emptyIndicesTexture,
    });

    this.rgbMinMaxLnScaleMinMax = new dyno.DynoVec4({
      value: new THREE.Vector4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX),
    });
    this.lodOpacity = new dyno.DynoBool({
      value: false,
    });

    this.dynoNumSh = new dyno.DynoInt({ value: 0 });
    this.shMax = new dyno.DynoVec3({ value: new THREE.Vector3() });

    let fileType = options.fileType;
    // lod-meta.json (PlayCanvas streamed SOG): solo da rootUrl
    const lodMeta =
      !options.fileBytes &&
      !options.fetchSource &&
      !!options.rootUrl &&
      (options.lodMeta ??
        (!fileType &&
          (isLodMetaUrl(options.rootUrl) || isSsogUrl(options.rootUrl))));
    if (options.fileBytes) {
      fileType ??= getSplatFileType(options.fileBytes);
      this.fetchSource = new FileBytesPagedSplatsFetcher(options.fileBytes);
    } else if (options.fetchSource) {
      this.fetchSource = options.fetchSource;
    } else if (options.rootUrl) {
      fileType ??= lodMeta
        ? SplatFileType.PCSOGS
        : getSplatFileTypeFromPath(options.rootUrl);
      this.fetchSource = new UrlPagedSplatsFetcher(
        options.rootUrl,
        options.requestHeader,
        options.withCredentials,
      );
    } else {
      throw new Error("No url, fileBytes or fetchSource provided");
    }

    if (!fileType) {
      throw new Error("Unable to determine file type");
    }
    this.fileType = fileType;

    if (lodMeta) {
      this.lodMetaPromise = this.getLodMeta();
      return;
    }
    if (this.fileType === SplatFileType.RAD) {
      this.radMetaPromise = this.getRadMeta();
    }
  }

  dispose() {
    if (this.dynoIndices.value !== PagedSplats.emptyIndicesTexture) {
      this.dynoIndices.value.dispose();
      this.dynoIndices.value = PagedSplats.emptyIndicesTexture;
    }

    this.abortController.abort();
  }

  setMaxSh(maxSh: number) {
    this.maxSh = maxSh;
  }

  getRadMeta(): Promise<{ meta: RadMeta; chunksStart: number }> {
    if (this.radMetaPromise) {
      return this.radMetaPromise;
    }

    this.radMetaPromise = (async () => {
      await wasm.initialization;

      // We don't know how big the header will be. Most likely 64KB will be enough,
      // but try larger blocks in backoff if it wasn't enough.
      for (const tryBytes of [65536, 256 * 1024, 1024 * 1024]) {
        const bytes = await this.fetchSource.fetchRange(
          this.abortController.signal,
          0,
          tryBytes,
        );
        const metaStart = decode_rad_header(bytes);
        if (metaStart) {
          return metaStart;
        }
        // Skip larger attempts when the current one already returned fewer bytes.
        if (bytes.length < tryBytes) {
          break;
        }
      }
      throw new Error("Failed to decode RAD header");
    })().then((metaStart) => {
      // console.log("RAD meta: ", metaStart.meta);
      return metaStart;
    });

    this.radMetaPromise.catch((error) => {
      // An AbortError can be expected in case the instance has been disposed.
      if (error.name !== "AbortError") {
        console.error(error);
      }
      // Allow it to be tried again
      // this.radMetaPromise = undefined;
    });

    return this.radMetaPromise;
  }

  getLodMeta(): Promise<LodMetaData> {
    if (this.lodMetaPromise) {
      return this.lodMetaPromise;
    }
    this.lodMetaPromise = (async () => {
      const fetcher = this.fetchSource as UrlPagedSplatsFetcher;
      const baseUrl = new URL(
        fetcher.rootUrl,
        globalThis.location.href,
      ).toString();
      let bytes: Uint8Array;
      let lodMetaUrl = baseUrl;
      if (isSsogUrl(baseUrl)) {
        // Archivio unico: i filenames di lod-meta.json diventano "<archivio>/<path entry>"
        const read = httpSsogRead(
          baseUrl,
          {
            headers: fetcher.requestHeader,
            credentials: fetcher.withCredentials ? "include" : "same-origin",
          },
          this.abortController.signal,
        );
        const index = await readSsogIndex(read);
        const prefix = `${baseUrl}/`;
        this.ssog = { read, index, prefix };
        lodMetaUrl = `${prefix}lod-meta.json`;
        bytes = await this.readSsog(lodMetaUrl);
      } else {
        bytes = await fetcher.fetchRangeImpl({
          url: baseUrl,
          requestHeader: fetcher.requestHeader,
          withCredentials: fetcher.withCredentials,
          signal: this.abortController.signal,
        });
      }
      this.lodMeta = parseLodMeta(
        JSON.parse(new TextDecoder().decode(bytes)),
        lodMetaUrl,
      );
      this.pager?.onUpdate?.();
      return this.lodMeta;
    })();
    this.lodMetaPromise.catch((error) => {
      this.lodMetaError = error;
      if (error.name !== "AbortError") {
        console.error(error);
      }
    });
    return this.lodMetaPromise;
  }

  // Entry dell'archivio .ssog a partire dall'url virtuale "<archivio>/<path>"
  readSsog(url: string): Promise<Uint8Array> {
    const ssog = this.ssog as NonNullable<typeof this.ssog>;
    const path = decodeURIComponent(url.slice(ssog.prefix.length));
    const entry = ssog.index.get(path);
    if (!url.startsWith(ssog.prefix) || !entry) {
      throw new Error(`ssog: entry "${path}" assente nell'archivio`);
    }
    return readSsogEntry(ssog.read, entry);
  }

  // Scarica meta.json + webp del file SOG (cartella o archivio .ssog) del chunk
  async fetchSogsFiles(chunk: number) {
    const { fileUrls } = await this.getLodMeta();
    if (chunk < 0 || chunk >= fileUrls.length) {
      throw new Error(
        `Chunk index out of range: ${chunk} (max: ${fileUrls.length - 1})`,
      );
    }
    const fetcher = this.fetchSource as UrlPagedSplatsFetcher;
    const fetchUrl = (url: string) =>
      this.ssog
        ? this.readSsog(url)
        : fetcher.fetchRangeImpl({
            url,
            requestHeader: fetcher.requestHeader,
            withCredentials: fetcher.withCredentials,
            signal: this.abortController.signal,
          });
    const metaUrl = fileUrls[chunk];
    const meta = await fetchUrl(metaUrl);
    const json = JSON.parse(new TextDecoder().decode(meta));
    if (json.count > 65536) {
      throw new Error(
        `lod-meta: "${metaUrl}" ha ${json.count} splat, max 65536 per file (1 file = 1 pagina)`,
      );
    }
    const names: string[] = [
      ...new Set<string>(
        ["means", "scales", "quats", "sh0", "shN"].flatMap(
          (key) => json[key]?.files ?? [],
        ),
      ),
    ];
    const files = await Promise.all(
      names.map((name) => fetchUrl(new URL(name, metaUrl).toString())),
    );
    return { meta, names, files };
  }

  async fetchDecodeChunk(chunk: number) {
    // Diagnostica [lod-diag]: fetch/decodifica di ogni chunk pagina (vedi desktop.log), per capire
    // dove si blocca la pipeline sui .ssog paged (stallo sul grezzo, caricamento a blocchi che poi
    // si ferma). Non throttled: un chunk fermo in fetch o decode senza mai loggare "end" è il sintomo.
    lodDiagLine("fetch start", chunk);
    const fetchStart = performance.now();
    let decodeBytes: Uint8Array | undefined = undefined;
    const sogs = this.lodMetaPromise
      ? await this.fetchSogsFiles(chunk)
      : undefined;

    if (sogs) {
      // decode sotto, nel worker
    } else if (this.fileType === SplatFileType.RAD) {
      const { meta, chunksStart } = await this.getRadMeta();
      if (chunk < 0 || chunk >= meta.chunks.length) {
        throw new Error(
          `Chunk index out of range: ${chunk} (max: ${meta.chunks.length - 1})`,
        );
      }
      let { offset, bytes, filename } = meta.chunks[chunk];

      if (filename) {
        decodeBytes = await this.fetchSource.fetchChunkFile(
          this.abortController.signal,
          filename,
        );
      } else {
        offset += chunksStart;
        // console.log(`Fetching chunk ${chunk} at offset ${offset} with bytes ${bytes}`);
        decodeBytes = await this.fetchSource.fetchRange(
          this.abortController.signal,
          offset,
          bytes,
        );
      }
    } else {
      decodeBytes = await this.fetchSource.fetchChunkById(
        this.abortController.signal,
        chunk,
      );
    }

    const fetchBytes = sogs
      ? sogs.meta.length + sogs.files.reduce((n, f) => n + f.length, 0)
      : (decodeBytes?.length ?? 0);
    lodDiagLine("fetch end", chunk, {
      ms: Math.round(performance.now() - fetchStart),
      bytes: fetchBytes,
    });

    lodDiagLine("decode start", chunk);
    const decodeStart = performance.now();
    try {
      const decodeResult = await this.decodeChunk(chunk, sogs, decodeBytes);
      lodDiagLine("decode end", chunk, {
        ms: Math.round(performance.now() - decodeStart),
      });
      return decodeResult;
    } catch (error) {
      lodDiagLine("decode error", chunk, {
        ms: Math.round(performance.now() - decodeStart),
        error: String(error),
      });
      throw error;
    }
  }

  private async decodeChunk(
    chunk: number,
    sogs: Awaited<ReturnType<PagedSplats["fetchSogsFiles"]>> | undefined,
    decodeBytes: Uint8Array | undefined,
  ) {
    return await workerPool.withWorker(async (worker) => {
      if (!this.pager) {
        throw new Error("PagedSplats.pager not set");
      }
      if (!this.pager.extSplats) {
        const result = sogs
          ? { lodSplats: await worker.call("decodeSogsPacked", sogs) }
          : await worker.call("loadPackedSplats", {
              fileBytes: decodeBytes,
              fileType: this.fileType,
              sh1Codes: this.sh1Codes?.slice(),
              sh2Codes: this.sh2Codes?.slice(),
              sh3Codes: (this.sh3Codes as Uint32Array | undefined)?.slice(),
            });
        if (!("lodSplats" in result)) {
          throw new Error("Loaded chunk does not contain LoD splats");
        }

        const lodSplats = result.lodSplats;
        if (!this.splatEncoding) {
          this.splatEncoding = lodSplats.splatEncoding;

          this.numSh = lodSplats.extra.sh3
            ? 3
            : lodSplats.extra.sh2
              ? 2
              : lodSplats.extra.sh1
                ? 1
                : 0;

          this.rgbMinMaxLnScaleMinMax.value.set(
            this.splatEncoding.rgbMin,
            this.splatEncoding.rgbMax,
            this.splatEncoding.lnScaleMin,
            this.splatEncoding.lnScaleMax,
          );

          this.lodOpacity.value = this.splatEncoding.lodOpacity;

          this.shMax.value.set(
            this.splatEncoding.sh1Max,
            this.splatEncoding.sh2Max,
            this.splatEncoding.sh3Max,
          );
        }
        this.sh1Codes = lodSplats.extra.sh1Codes ?? this.sh1Codes;
        this.sh2Codes = lodSplats.extra.sh2Codes ?? this.sh2Codes;
        this.sh3Codes = lodSplats.extra.sh3Codes ?? this.sh3Codes;
        return lodSplats;
      }

      const sh3Codes = this.sh3Codes as [Uint32Array, Uint32Array] | undefined;
      const result = sogs
        ? { lodSplats: await worker.call("decodeSogsExt", sogs) }
        : await worker.call("loadExtSplats", {
            fileBytes: decodeBytes,
            fileType: this.fileType,
            sh1Codes: this.sh1Codes?.slice(),
            sh2Codes: this.sh2Codes?.slice(),
            sh3Codes: sh3Codes
              ? [sh3Codes[0].slice(), sh3Codes[1].slice()]
              : undefined,
          });
      if (!("lodSplats" in result)) {
        throw new Error("Loaded chunk does not contain LoD splats");
      }
      const lodSplats = result.lodSplats;
      if (!this.splatEncoding) {
        this.splatEncoding = DEFAULT_SPLAT_ENCODING;
        this.numSh =
          lodSplats.extra.sh3a && lodSplats.extra.sh3b
            ? 3
            : lodSplats.extra.sh2
              ? 2
              : lodSplats.extra.sh1
                ? 1
                : 0;
      }
      this.sh1Codes = lodSplats.extra.sh1Codes ?? this.sh1Codes;
      this.sh2Codes = lodSplats.extra.sh2Codes ?? this.sh2Codes;
      this.sh3Codes = lodSplats.extra.sh3Codes ?? this.sh3Codes;
      return lodSplats;
    });
  }

  /** Draw nothing until the next update(), keeping the indices texture allocated. */
  clear() {
    this.numSplats = 0;
    this.dynoNumSplats.value = 0;
  }

  update(numSplats: number, indices: Uint32Array) {
    if (!this.pager) {
      throw new Error("PagedSplats.pager not set");
    }

    const renderer = this.pager.renderer;
    this.numSplats = numSplats;
    this.indicesVersion++;
    this.dynoNumSplats.value = this.numSplats;
    const rows = Math.ceil(numSplats / 16384);

    let indicesTexture =
      this.dynoIndices.value === PagedSplats.emptyIndicesTexture
        ? undefined
        : this.dynoIndices.value;
    if (indicesTexture && rows > indicesTexture.image.height) {
      indicesTexture.dispose();
      indicesTexture = undefined;
    }

    if (!indicesTexture) {
      indicesTexture = new THREE.DataTexture(
        indices,
        4096,
        rows,
        THREE.RGBAIntegerFormat,
        THREE.UnsignedIntType,
      );
      indicesTexture.internalFormat = "RGBA32UI";
      indicesTexture.needsUpdate = true;
      renderer?.initTexture(indicesTexture);
      this.dynoIndices.value = indicesTexture;
    } else {
      const textureIndices = indicesTexture.image.data as Uint32Array;
      textureIndices.set(indices.subarray(0, numSplats));

      // The native WebGPU path reads the CPU indices directly.
      if (renderer)
        uploadU32DataTextureRows(
          renderer,
          indicesTexture,
          4096,
          rows,
          textureIndices,
        );
    }
  }

  prepareFetchSplat() {}

  getNumSplats(): number {
    return this.numSplats;
  }

  hasRgbDir(): boolean {
    if (!this.pager) {
      return false;
    }
    return Math.min(this.numSh, this.maxSh, this.pager.maxSh) > 0;
  }

  getNumSh(): number {
    return this.numSh;
  }

  fetchSplat({
    index,
    viewOrigin,
  }: {
    index: dyno.DynoVal<"int">;
    viewOrigin?: dyno.DynoVal<"vec3">;
  }): dyno.DynoVal<typeof dyno.Gsplat> {
    if (!this.pager) {
      throw new Error("PagedSplats.pager not set");
    }

    const splatIndex = this.pager.readIndex.apply({
      index,
      numSplats: this.dynoNumSplats,
      indices: this.dynoIndices,
    }).index;

    if (!this.pager.extSplats) {
      if (this.hasRgbDir() && viewOrigin) {
        this.dynoNumSh.value = Math.min(
          this.numSh,
          this.maxSh,
          this.pager.maxSh,
        );
        return this.pager.readSplatDir.apply({
          index: splatIndex,
          rgbMinMaxLnScaleMinMax: this.rgbMinMaxLnScaleMinMax,
          lodOpacity: this.lodOpacity,
          viewOrigin,
          numSh: this.dynoNumSh,
          shMax: this.shMax,
        }).gsplat;
      }
      return this.pager.readSplat.apply({
        index: splatIndex,
        rgbMinMaxLnScaleMinMax: this.rgbMinMaxLnScaleMinMax,
        lodOpacity: this.lodOpacity,
      }).gsplat;
    }

    if (this.hasRgbDir() && viewOrigin) {
      this.dynoNumSh.value = Math.min(this.numSh, this.maxSh, this.pager.maxSh);
      return this.pager.readSplatExtDir.apply({
        index: splatIndex,
        viewOrigin,
        numSh: this.dynoNumSh,
      }).gsplat;
    }
    return this.pager.readSplatExt.apply({ index: splatIndex }).gsplat;
  }

  // Iterate over Gsplats index 0..=(this.numSplats-1), unpack each Gsplat
  // and invoke the callback function with the Gsplat attributes.
  forEachSplat(
    callback: (
      index: number,
      center: THREE.Vector3,
      scales: THREE.Vector3,
      quaternion: THREE.Quaternion,
      opacity: number,
      color: THREE.Color,
    ) => void,
  ) {
    if (!this.pager || !this.numSplats) {
      return;
    }
    const extSplats = this.pager.extSplats;
    const indices = this.dynoIndices.value.image.data as Uint32Array;
    const packedSplatArray = this.pager.packedTexture.value.image
      .data as Uint32Array;
    const extPackedSplatArray = this.pager.extTexture.value.image
      .data as Uint32Array;
    const extArrays: [Uint32Array, Uint32Array] = [
      packedSplatArray,
      extPackedSplatArray,
    ];

    for (let i = 0; i < this.numSplats; ++i) {
      const splatIndex = indices[i];
      const unpacked = extSplats
        ? decodeExtSplat(extArrays, splatIndex)
        : unpackSplat(packedSplatArray, splatIndex, this.splatEncoding);
      callback(
        i,
        unpacked.center,
        unpacked.scales,
        unpacked.quaternion,
        unpacked.opacity,
        unpacked.color,
      );
    }
  }

  static emptyIndicesTexture = (() => {
    const emptyArray = new Uint32Array(4096 * 4);
    const texture = new THREE.DataTexture(emptyArray, 4096, 1);
    texture.format = THREE.RGBAIntegerFormat;
    texture.type = THREE.UnsignedIntType;
    texture.internalFormat = "RGBA32UI";
    texture.needsUpdate = true;
    return texture;
  })();
}
