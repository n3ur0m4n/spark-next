import type * as THREE from "three";

// LoD "streamed SOG" di PlayCanvas (lod-meta.json, splat-transform):
// albero k-d di foglie, ogni foglia ha un range contiguo [offset, offset+count)
// in un file SOG per ciascun livello (0 = più dettagliato). 1 file = 1 chunk/pagina
// del SplatPager. Per ogni foglia si disegna UN livello (sostituzione, niente blending).
// Selezione e streaming replicano gsplat-octree-instance / gsplat-lod-table di PlayCanvas.

const PAGE_SPLATS = 65536;
const LOD_BASE = 5; // lodBaseDistance PlayCanvas
const LN_LOD_MULT = Math.log(3); // lodMultiplier = 3
const BEHIND_PENALTY = 3; // lodBehindPenalty (valore esempio PC)
const DISTANCE_SHRINK = 0.75; // lodDistanceShrink PC
const UNDERFILL_LIMIT = 5; // lodUnderfillLimit (valore esempio PC)
const REF_TAN_HALF_FOV = Math.tan((22.5 * Math.PI) / 180);

// Tier di caricamento PC (gsplat-octree-instance.js:31-33) + 3 = file già mostrato
const TIER_PREFETCH = 0;
const TIER_SWITCH = 1;
const TIER_VISIBLE = 2;
const TIER_SHOWN = 3;

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
  table?: { rangeMin: number; bandLod: Int16Array; bandCount: Uint32Array };
}

type LodMetaNode = {
  bound: { min: number[]; max: number[] };
  children?: LodMetaNode[];
  lods?: Record<string, { file: number; offset: number; count: number }>;
};

export function isLodMetaUrl(url: string): boolean {
  return /lod-meta\.json$/i.test(url.split(/[?#]/)[0]);
}

/** Compensazione FOV PC: min(tanHalfV, tanHalfH) / tan(22.5°) */
export function lodMetaFovScale(tanHalfVFov: number, aspect: number): number {
  const a = aspect > 0 && Number.isFinite(aspect) ? aspect : 1;
  return Math.min(tanHalfVFov, tanHalfVFov * a) / REF_TAN_HALF_FOV;
}

export function parseLodMeta(
  json: {
    lodLevels: number;
    filenames: string[];
    tree: LodMetaNode;
  },
  baseUrl: string,
): LodMetaData {
  const numLevels = json.lodLevels;
  const fileUrls = json.filenames.map((f) => new URL(f, baseUrl).toString());

  const leaves: LodMetaNode[] = [];
  const stack = [json.tree];
  while (stack.length > 0) {
    const node = stack.pop() as LodMetaNode;
    if (node.children) {
      stack.push(...node.children);
    } else if (node.lods && Object.keys(node.lods).length > 0) {
      leaves.push(node);
    }
  }

  const numLeaves = leaves.length;
  const bounds = new Float32Array(numLeaves * 6);
  const file = new Int32Array(numLeaves * numLevels).fill(-1);
  const offset = new Uint32Array(numLeaves * numLevels);
  const count = new Uint32Array(numLeaves * numLevels);
  const fileLevel = new Int32Array(fileUrls.length);

  for (let i = 0; i < numLeaves; i++) {
    const { bound, lods } = leaves[i];
    bounds.set(bound.min, i * 6);
    bounds.set(bound.max, i * 6 + 3);
    for (const [level, lod] of Object.entries(lods ?? {})) {
      const l = Number(level);
      if (!(l >= 0 && l < numLevels) || !(lod.count > 0)) continue;
      if (lod.offset + lod.count > PAGE_SPLATS) {
        throw new Error(
          `lod-meta: range ${lod.offset}+${lod.count} oltre ${PAGE_SPLATS} splat in "${json.filenames[lod.file]}" (1 file deve stare in una pagina)`,
        );
      }
      file[i * numLevels + l] = lod.file;
      offset[i * numLevels + l] = lod.offset;
      count[i * numLevels + l] = lod.count;
      fileLevel[lod.file] = l;
    }
    // Livelli mancanti restano -1: buchi e livello vuoto risolti in lodTable() come PC
  }

  // PC GSplatOctree._computeBoundsExcess: eccesso sulla semi-estensione mediana per asse
  const excess = new Float32Array(numLeaves * 3);
  if (numLeaves > 0) {
    const half = new Float32Array(numLeaves);
    for (let axis = 0; axis < 3; axis++) {
      for (let i = 0; i < numLeaves; i++) {
        half[i] = (bounds[i * 6 + 3 + axis] - bounds[i * 6 + axis]) * 0.5;
      }
      const typical = half.slice().sort()[numLeaves >> 1];
      for (let i = 0; i < numLeaves; i++) {
        excess[i * 3 + axis] = Math.max(0, half[i] - typical);
      }
    }
  }

  return {
    fileUrls,
    numLeaves,
    numLevels,
    bounds,
    excess,
    file,
    offset,
    count,
    fileLevel,
    rangeMin: numLevels - 1,
    current: new Int16Array(numLeaves).fill(-1),
  };
}

// PC GSplatLodTable (gsplat-lod-table.js:101-171), range [rangeMin, numLevels-1]:
// oltre il livello più grossolano con dati = livello vuoto (0 splat, niente file);
// buco = livello più fine con dati, altrimenti il più grossolano.
function lodTable(data: LodMetaData) {
  const { numLeaves, numLevels: K, rangeMin, file, count } = data;
  if (data.table?.rangeMin === rangeMin) return data.table;
  const rangeMax = K - 1;
  const span = rangeMax - rangeMin + 1;
  const bandLod = new Int16Array(numLeaves * span).fill(-1);
  const bandCount = new Uint32Array(numLeaves * span);
  for (let i = 0; i < numLeaves; i++) {
    const base = i * K;
    const row = i * span;
    let coarsest = -1;
    for (let l = rangeMax; l >= rangeMin && coarsest < 0; l--) {
      if (file[base + l] >= 0) coarsest = l;
    }
    if (coarsest < 0) continue;
    for (let b = 0; b < span; b++) {
      const band = rangeMin + b;
      let lod = band;
      if (band > coarsest) {
        lod = coarsest + 1;
      } else if (file[base + band] < 0) {
        lod = -1;
        for (let l = band - 1; l >= rangeMin && lod < 0; l--) {
          if (file[base + l] >= 0) lod = l;
        }
        for (let l = band + 1; l <= coarsest && lod < 0; l++) {
          if (file[base + l] >= 0) lod = l;
        }
      }
      bandLod[row + b] = lod;
      bandCount[row + b] = lod > coarsest ? 0 : count[base + lod];
    }
  }
  data.table = { rangeMin, bandLod, bandCount };
  return data.table;
}

/**
 * Distanza mondo camera → foglia come PC evaluateNodeDistances: punto più vicino
 * dell'AABB ristretto (lodDistanceShrink), penalità dietro camera, × scala mondo
 * del mesh × compensazione FOV.
 */
export function lodMetaDistances(
  data: LodMetaData,
  viewToObject: THREE.Matrix4,
  fovScale = 1,
): Float64Array {
  const e = viewToObject.elements;
  const cx = e[12];
  const cy = e[13];
  const cz = e[14];
  // la camera ha scala 1: |colonna| di viewToObject = 1 / scala mondo del mesh
  const scale = 1 / Math.hypot(e[0], e[1], e[2]);
  // asse -Z della camera nello spazio locale, normalizzato
  const fl = Math.hypot(e[8], e[9], e[10]);
  const fx = -e[8] / fl;
  const fy = -e[9] / fl;
  const fz = -e[10] / fl;
  const penalty = BEHIND_PENALTY - 1;
  const distanceScale = fovScale * scale;
  const b = data.bounds;
  const ex = data.excess;
  const out = new Float64Array(data.numLeaves);
  for (let i = 0; i < data.numLeaves; i++) {
    const j = i * 6;
    const sx = ex[i * 3] * DISTANCE_SHRINK;
    const sy = ex[i * 3 + 1] * DISTANCE_SHRINK;
    const sz = ex[i * 3 + 2] * DISTANCE_SHRINK;
    const px = Math.min(Math.max(cx, b[j] + sx), b[j + 3] - sx);
    const py = Math.min(Math.max(cy, b[j + 1] + sy), b[j + 4] - sy);
    const pz = Math.min(Math.max(cz, b[j + 2] + sz), b[j + 5] - sz);
    const dx = px - cx;
    const dy = py - cy;
    const dz = pz - cz;
    const dSq = dx * dx + dy * dy + dz * dz;
    let factor = 1;
    if (dSq > 0.0001) {
      const dot = fx * dx + fy * dy + fz * dz;
      // 0 di lato → 1 dietro
      if (dot < 0) factor = 1 + (-dot / Math.sqrt(dSq)) * penalty;
    }
    out[i] = Math.sqrt(dSq) * factor * distanceScale;
  }
  return out;
}

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
  priority: { splats: S; chunk: number }[];
  /** prefetch del passo successivo verso l'ottimo (priorità più bassa) */
  prefetch: { splats: S; chunk: number }[];
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
export function selectLodMeta<S>({
  instances,
  maxSplats,
  maxFiles,
  residentPage,
  indicesCache,
  fovScale = 1,
  lodLimit = false,
}: {
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
}): LodMetaResult<S> {
  const tables = instances.map(({ data }) => lodTable(data));
  const dists = instances.map(({ data, viewToObject }) =>
    lodMetaDistances(data, viewToObject, fovScale),
  );
  // a[i] = 1 + log_3(d_i / (b·lodScale)) → banda = floor(a - c), c = log_3(s)
  const as = instances.map(({ lodScale }, m) => {
    const bias = Math.log(LOD_BASE * lodScale);
    return dists[m].map(
      (d) => 1 + (Math.log(Math.max(d, 1e-6)) - bias) / LN_LOD_MULT,
    );
  });

  const stamps = instances.map(
    ({ data }) => new Int32Array(data.fileUrls.length),
  );
  let stamp = 0;
  const cost = (c: number) => {
    stamp++;
    let splats = 0;
    let files = 0;
    for (let m = 0; m < instances.length; m++) {
      const { data } = instances[m];
      const { bandLod, bandCount } = tables[m];
      const a = as[m];
      const st = stamps[m];
      const K = data.numLevels;
      const rMin = data.rangeMin;
      const span = K - rMin;
      for (let i = 0; i < data.numLeaves; i++) {
        const b =
          i * span +
          Math.min(Math.max(Math.floor(a[i] - c), rMin), K - 1) -
          rMin;
        const lod = bandLod[b];
        if (lod < 0) continue;
        splats += bandCount[b];
        const f = data.file[i * K + lod];
        if (f >= 0 && st[f] !== stamp) {
          st[f] = stamp;
          files++;
        }
      }
    }
    return { splats, files };
  };
  const fits = (c: number) => {
    const { splats, files } = cost(c);
    return splats <= maxSplats && files <= maxFiles;
  };

  // Ricerca binaria sulla c più grande (s più grande = più dettaglio) che sta nel budget
  let minA = Number.POSITIVE_INFINITY;
  let maxA = Number.NEGATIVE_INFINITY;
  let maxK = 1;
  for (let m = 0; m < instances.length; m++) {
    maxK = Math.max(maxK, instances[m].data.numLevels);
    for (const v of as[m]) {
      minA = Math.min(minA, v);
      maxA = Math.max(maxA, v);
    }
  }
  let c = maxA + 1; // tutto al livello più fine
  if (lodLimit) c = Math.min(c, 0); // dettaglio max c=0, il budget può solo abbassarlo
  if (instances.length > 0 && !fits(c)) {
    let lo = minA - maxK; // tutto al livello più grossolano
    let hi = c;
    if (fits(lo)) {
      for (let iter = 0; iter < 32; iter++) {
        const mid = 0.5 * (lo + hi);
        if (fits(mid)) lo = mid;
        else hi = mid;
      }
    }
    // ponytail: se nemmeno il più grossolano sta nel budget si mostra quello comunque
    c = lo;
  }

  const numSplats: number[] = [];
  const indices: Uint32Array[] = [];
  const entries: { splats: S; chunk: number; prio: number }[] = [];
  const prefetchEntries: { splats: S; chunk: number; prio: number }[] = [];
  let rangeChanged = false;

  for (let m = 0; m < instances.length; m++) {
    const { splats, data } = instances[m];
    const { bandLod } = tables[m];
    const a = as[m];
    const dist = dists[m];
    const K = data.numLevels;
    const rMin = data.rangeMin;
    const span = K - rMin;
    const numFiles = data.fileUrls.length;
    const pageOf = new Int32Array(numFiles).fill(-2);
    const getPage = (f: number) => {
      if (pageOf[f] === -2) pageOf[f] = residentPage(splats, f);
      return pageOf[f];
    };
    // priorità PC requestFile: tier + (livello file + distanceRank) / livelli, la più alta vince
    const prios = new Float64Array(numFiles).fill(-1);
    let pending = 0;
    const request = (f: number, tier: number, rank: number) => {
      if (tier !== TIER_SHOWN) {
        if (getPage(f) >= 0) return;
        pending++;
      }
      const p = tier + (data.fileLevel[f] + rank) / K;
      if (p > prios[f]) prios[f] = p;
    };

    let total = 0;
    for (let i = 0; i < data.numLeaves; i++) {
      const row = i * span;
      const fileOf = (lod: number) => (lod < 0 ? -1 : data.file[i * K + lod]);
      // catena PC: livelli distinti delle bande, dal più fine al più grossolano
      const coarserOnChain = (lod: number) => {
        for (let b = row; b < row + span; b++) {
          if (bandLod[b] > lod) return bandLod[b];
        }
        return -1;
      };
      const finerOnChain = (lod: number) => {
        for (let b = row + span - 1; b >= row; b--) {
          if (bandLod[b] >= 0 && bandLod[b] < lod) return bandLod[b];
        }
        return -1;
      };

      const optimal =
        bandLod[
          row + Math.min(Math.max(Math.floor(a[i] - c), rMin), K - 1) - rMin
        ];

      // PC selectDesiredLodIndex: il più fine già caricato entro UNDERFILL_LIMIT passi
      // (il livello vuoto conta come caricato), altrimenti il più grossolano con file
      let desired = optimal;
      if (optimal >= 0) {
        let loaded = -1;
        let lod = optimal;
        for (let step = 0; step <= UNDERFILL_LIMIT && lod >= 0; step++) {
          const f = fileOf(lod);
          if (f < 0 || getPage(f) >= 0) {
            loaded = lod;
            break;
          }
          lod = coarserOnChain(lod);
        }
        if (loaded >= 0) {
          desired = loaded;
        } else {
          let coarsest = -1;
          lod = optimal;
          for (let step = 0; step <= UNDERFILL_LIMIT && lod >= 0; step++) {
            if (fileOf(lod) >= 0) coarsest = lod;
            lod = coarserOnChain(lod);
          }
          if (coarsest >= 0) desired = coarsest;
        }
      }

      // PC applyLodChanges: il livello mostrato cambia solo quando il voluto è
      // residente; vuoto = sparisce subito
      const rank = 1 / (1 + Math.max(dist[i] * dist[i], 1e-12));
      const desiredFile = fileOf(desired);
      let cur = data.current[i];
      if (desiredFile < 0) {
        cur = -1;
      } else if (getPage(desiredFile) >= 0) {
        cur = desired;
      } else if (cur >= 0 && getPage(fileOf(cur)) >= 0) {
        request(desiredFile, TIER_SWITCH, rank);
      } else {
        cur = -1;
        request(desiredFile, TIER_VISIBLE, rank);
      }
      data.current[i] = cur;

      // PC prefetchNextLod: solo il passo successivo verso l'ottimo
      if (desired >= 0 && optimal >= 0) {
        const target = desired === optimal ? optimal : finerOnChain(desired);
        const f = fileOf(target);
        if (f >= 0) request(f, TIER_PREFETCH, rank);
      }

      if (cur >= 0) {
        request(fileOf(cur), TIER_SHOWN, rank);
        total += data.count[i * K + cur];
      }
    }

    // Bootstrap PC (esempio a frame:ready): tutto il richiesto caricato → range completo
    if (rMin > 0 && pending === 0) {
      data.rangeMin = 0;
      rangeChanged = true;
    }

    const capacity = Math.max(1, Math.ceil(total / 16384)) * 16384;
    let out = indicesCache?.[m];
    if (!out || out.length < capacity) {
      out = new Uint32Array(Math.ceil((capacity * 1.25) / 16384) * 16384);
    }
    let n = 0;
    for (let i = 0; i < data.numLeaves; i++) {
      const cur = data.current[i];
      if (cur < 0) continue;
      const k = i * K + cur;
      const base = getPage(data.file[k]) * PAGE_SPLATS + data.offset[k];
      const cnt = data.count[k];
      for (let j = 0; j < cnt; j++) out[n + j] = base + j;
      n += cnt;
    }
    numSplats.push(n);
    indices.push(out);

    for (let f = 0; f < numFiles; f++) {
      const prio = prios[f];
      if (prio >= TIER_SWITCH) entries.push({ splats, chunk: f, prio });
      else if (prio >= 0) prefetchEntries.push({ splats, chunk: f, prio });
    }
  }

  entries.sort((x, y) => y.prio - x.prio);
  prefetchEntries.sort((x, y) => y.prio - x.prio);
  const strip = ({ splats, chunk }: { splats: S; chunk: number }) => ({
    splats,
    chunk,
  });
  return {
    numSplats,
    indices,
    priority: entries.map(strip),
    prefetch: prefetchEntries.map(strip),
    rangeChanged,
  };
}
