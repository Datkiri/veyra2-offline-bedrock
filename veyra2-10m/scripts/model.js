import { CHUNKS, TENSORS, TOTAL_BYTES, WEIGHTS_ID } from "./generated/model_data.js";
import { PREFIX_CHUNKS, PREFIX_META } from "./generated/prefix_data.js";
import { DIM } from "./generated/model_config.js";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const DECODE = new Int16Array(128);
DECODE.fill(-1);
for (let i = 0; i < ALPHABET.length; i++) DECODE[ALPHABET.charCodeAt(i)] = i;

function* decodeChunks(chunks, totalBytes, workPerYield) {
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  let work = 0;
  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
    const chunk = chunks[chunkIndex];
    if (typeof chunk !== "string") throw new Error(`Missing Veyra2 weight chunk ${chunkIndex}`);
    if (chunk.length % 4 !== 0) throw new Error(`Invalid base64 length in chunk ${chunkIndex}`);
    for (let i = 0; i < chunk.length; i += 4) {
      const a = DECODE[chunk.charCodeAt(i)];
      const b = DECODE[chunk.charCodeAt(i + 1)];
      const cCode = chunk.charCodeAt(i + 2);
      const dCode = chunk.charCodeAt(i + 3);
      const c = cCode === 61 ? 0 : DECODE[cCode];
      const d = dCode === 61 ? 0 : DECODE[dCode];
      bytes[offset++] = (a << 2) | (b >> 4);
      if (cCode !== 61) bytes[offset++] = ((b & 15) << 4) | (c >> 2);
      if (dCode !== 61) bytes[offset++] = ((c & 3) << 6) | d;
      work += 4;
      if (work >= workPerYield) {
        work = 0;
        yield;
      }
    }
    chunks[chunkIndex] = null;
    yield;
  }
  if (offset !== totalBytes) throw new Error(`Veyra2 byte count mismatch: ${offset}/${totalBytes}`);
  return bytes;
}
export function* loadModel(options = {}) {
  const workPerYield = options.workPerYield ?? 262144;
  const bytes = yield* decodeChunks(CHUNKS, TOTAL_BYTES, workPerYield);
  const tensors = {};
  for (const [name, spec] of Object.entries(TENSORS)) {
    if (spec.kind === "q8") {
      const [rows, cols] = spec.shape;
      tensors[name] = {
        kind: "q8",
        layout: spec.layout || "row",
        rows,
        cols,
        values: new Int8Array(bytes.buffer, spec.values, rows * cols),
        scales: new Float32Array(bytes.buffer, spec.scales, rows),
      };
    } else {
      tensors[name] = new Float32Array(bytes.buffer, spec.values, spec.shape[0]);
    }
  }
  Object.defineProperty(tensors, "weightsId", { value: WEIGHTS_ID });
  if (PREFIX_META && options.prefixCache !== false) {
    const meta = PREFIX_META;
    const valid = meta.modelId === WEIGHTS_ID && meta.layers === DIM.layers &&
      meta.kvWidth === DIM.kvHeads * DIM.headSize && meta.hiddenSize === DIM.hidden &&
      Array.isArray(meta.ids) && meta.ids.length > 0;
    if (!valid) {
      console.warn("[Veyra2] Prefix snapshot differs from model; computing prompt normally.");
    } else {
      const prefix = yield* decodeChunks(PREFIX_CHUNKS, meta.bytes, workPerYield);
      const width = meta.ids.length * meta.kvWidth;
      const keys = [], values = [];
      for (let layer = 0; layer < DIM.layers; layer++) {
        keys.push(new Float32Array(prefix.buffer, layer * width * 8, width));
        values.push(new Float32Array(prefix.buffer, layer * width * 8 + width * 4, width));
      }
      Object.defineProperty(tensors, "prefixSeed", { value: {
        ids: meta.ids, keys, values, bytes: meta.bytes,
        hidden: new Float32Array(prefix.buffer, DIM.layers * width * 8, DIM.hidden),
      } });
    }
  }
  return tensors;
}
