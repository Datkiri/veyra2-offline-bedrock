                                                                                     
const selected = Object.create(null);
let lastReport = null;
const SHAPES = ["model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.q_proj.weight",
  "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight"];
const clock = () => typeof globalThis.performance?.now === "function" ? performance.now() : Date.now();
export const maxRowsPerSlice = () => 32;
export const shapeOf = matrix => `${matrix.cols}x${matrix.rows}`;
export const getKernelMode = matrix => selected[shapeOf(matrix)] || "scalar";
export const getKernelStatus = () => ({ selected: { ...selected }, lastReport, evalAvailable: false });
export const withinTolerance = (a, b) => Number.isFinite(a) && Number.isFinite(b) &&
  Math.abs(a - b) <= 1e-5 + 1e-5 * Math.max(Math.abs(a), Math.abs(b));
export function demoteKernel(matrix, reason) {
  selected[shapeOf(matrix)] = "scalar";
  console.warn(`[Veyra2] ${shapeOf(matrix)} reverted to scalar: ${reason}`);
}
export function scalarRows(input, matrix, output, start, end) {
  const { values, scales, cols } = matrix;
  const stride = matrix.layout === "tile4" ? 4 : 1;
  for (let row = start; row < end; row++) {
    let sum = 0;
    let index = stride === 4 ? (row - (row & 3)) * cols + (row & 3) : row * cols;
    for (let col = 0; col < cols; col++, index += stride) sum += input[col] * values[index];
    output[row] = sum * scales[row];
  }
}
function unrolledRows(input, matrix, output, start, end) {
  const { values, scales, cols } = matrix;
  const stride = matrix.layout === "tile4" ? 4 : 1;
  for (let row = start; row < end; row++) {
    let index = stride === 4 ? (row - (row & 3)) * cols + (row & 3) : row * cols;
    let a = 0, b = 0, c = 0, d = 0, col = 0;
    for (; col + 3 < cols; col += 4, index += 4 * stride) {
      a += input[col] * values[index];
      b += input[col + 1] * values[index + stride];
      c += input[col + 2] * values[index + 2 * stride];
      d += input[col + 3] * values[index + 3 * stride];
    }
    let sum = (a + b) + (c + d);
    for (; col < cols; col++, index += stride) sum += input[col] * values[index];
    output[row] = sum * scales[row];
  }
}
function tiledRows(input, matrix, output, start, end) {
  if (matrix.layout !== "tile4") return scalarRows(input, matrix, output, start, end);
  const { values, scales, cols } = matrix;
  let row = start;
  const aligned = Math.min(end, (row + 3) & ~3);
  scalarRows(input, matrix, output, row, aligned);
  row = aligned;
  for (; row + 3 < end; row += 4) {
    let a = 0, b = 0, c = 0, d = 0;
    let index = row * cols;
    for (let col = 0; col < cols; col++, index += 4) {
      const x = input[col];
      a += x * values[index]; b += x * values[index + 1];
      c += x * values[index + 2]; d += x * values[index + 3];
    }
    output[row] = a * scales[row]; output[row + 1] = b * scales[row + 1];
    output[row + 2] = c * scales[row + 2]; output[row + 3] = d * scales[row + 3];
  }
  scalarRows(input, matrix, output, row, end);
}
export function runKernelRows(mode, input, matrix, output, start, end) {
  (mode === "tile4" ? tiledRows : mode === "unrolled" ? unrolledRows : scalarRows)(input, matrix, output, start, end);
}
function* measure(mode, input, matrix, output, rows, repetitions, deadline) {
  let elapsed = 0, passes = 0;
  for (let pass = 0; pass < repetitions && clock() < deadline; pass++) {
    for (let row = 0; row < rows; row += 32) {
      const started = clock();
      runKernelRows(mode, input, matrix, output, row, Math.min(rows, row + 32));
      elapsed += clock() - started;
      yield;
    }
    passes++;
  }
  return { ms: passes ? elapsed / passes : Infinity, passes };
}
function* benchmark(tensors, options, full) {
  const started = clock();
  const deadline = started + (options.maxMs ?? (full ? 30000 : 5000));
  const results = {}, verified = {};
  for (const name of SHAPES) {
    if (clock() >= deadline) break;
    const matrix = tensors[name], shape = shapeOf(matrix);
    const rows = full ? matrix.rows : Math.min(256, matrix.rows);
    const input = Float32Array.from({ length: matrix.cols }, (_, i) => Math.sin(i * 0.17));
    const reference = new Float32Array(rows), output = new Float32Array(rows), tileOutput = new Float32Array(rows);
    for (let row = 0; row < rows; row += 32) {
      scalarRows(input, matrix, reference, row, Math.min(rows, row + 32));
      unrolledRows(input, matrix, output, row, Math.min(rows, row + 32));
      tiledRows(input, matrix, tileOutput, row, Math.min(rows, row + 32));
      yield;
    }
    let valid = true, tileValid = true;
    for (let row = 0; row < rows; row++) {
      valid &&= withinTolerance(reference[row], output[row]);
      tileValid &&= withinTolerance(reference[row], tileOutput[row]);
    }
    const scalar = yield* measure("scalar", input, matrix, output, rows, full ? 4 : 2, deadline);
    const unrolled = yield* measure("unrolled", input, matrix, output, rows, full ? 4 : 2, deadline);
    const tiled = yield* measure("tile4", input, matrix, output, rows, full ? 4 : 2, deadline);
    selected[shape] = valid && scalar.passes && unrolled.passes && unrolled.ms < scalar.ms * 0.95 ? "unrolled" : "scalar";
    const best = selected[shape] === "unrolled" ? unrolled : scalar;
    if (tileValid && tiled.passes && (!best.passes || tiled.ms < best.ms * 0.95 ||
      (tiled.ms === 0 && best.ms === 0))) selected[shape] = "tile4";
    results[shape] = { rows, scalarMs: scalar.ms, unrolledMs: unrolled.ms, tile4Ms: tiled.ms, selected: selected[shape] };
    verified[shape] = { scalar: true, unrolled: valid, tile4: tileValid };
  }
  lastReport = { elapsedMs: clock() - started, selected: { ...selected }, results, verified };
  return lastReport;
}
export function* calibrateKernels(tensors, options = {}) { return yield* benchmark(tensors, options, false); }
export function* benchKernels(tensors, options = {}) { return yield* benchmark(tensors, options, true); }
