import { compileKernel, getDynamicStatus } from "./dynamic.js";
import { calibrateMathKernels, getMathStatus } from "./math_kernels.js";

const selected = Object.create(null);
const generated = new Map();
let lastReport = null;
const MODES = ["scalar", "unrolled", "tile4", "function4", "function8", "function16", "eval4", "eval8", "eval16"];
const SHAPES = ["model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.q_proj.weight",
  "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight"];
const clock = () => typeof globalThis.performance?.now === "function" ? performance.now() : Date.now();
export const maxRowsPerSlice = () => 32;
export const shapeOf = matrix => `${matrix.cols}x${matrix.rows}`;
export const getKernelMode = matrix => selected[shapeOf(matrix)] || "scalar";
export const getKernelStatus = () => {
  const dynamic = getDynamicStatus();
  return { selected: { ...selected }, lastReport, dynamic, math: getMathStatus(),
    evalAvailable: dynamic.engines.eval.available, functionAvailable: dynamic.engines.function.available };
};
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
function sourceFor(cols, unroll) {
  const edge = `let sum=0,index=(row-(row&3))*${cols}+(row&3);for(let col=0;col<${cols};col++,index+=4)sum+=input[col]*values[index];output[row]=sum*scales[row];`;
  const limit = cols - cols % unroll;
  let block = "";
  for (let i = 0; i < unroll; i++) {
    const col = i ? `col+${i}` : "col", base = i * 4;
    block += `const x${i}=input[${col}];a+=x${i}*values[index+${base}];b+=x${i}*values[index+${base + 1}];c+=x${i}*values[index+${base + 2}];d+=x${i}*values[index+${base + 3}];`;
  }
  return `if(matrix.cols!==${cols}||matrix.layout!=="tile4")throw new Error("Generated matrix layout mismatch");
    const values=matrix.values,scales=matrix.scales;let row=start;
    const aligned=Math.min(end,(row+3)&~3);
    for(;row<aligned;row++){${edge}}
    for(;row+3<end;row+=4){
      let a=0,b=0,c=0,d=0,index=row*${cols},col=0;
      for(;col<${limit};col+=${unroll},index+=${unroll * 4}){${block}}
      for(;col<${cols};col++,index+=4){const x=input[col];a+=x*values[index];b+=x*values[index+1];c+=x*values[index+2];d+=x*values[index+3];}
      output[row]=a*scales[row];output[row+1]=b*scales[row+1];
      output[row+2]=c*scales[row+2];output[row+3]=d*scales[row+3];
    }
    for(;row<end;row++){${edge}}`;
}

function generatedFor(mode, cols) {
  const key = `${mode}:${cols}`;
  if (generated.has(key)) return generated.get(key);
  const match = /^(function|eval)(4|8|16)$/.exec(mode);
  if (!match || (cols !== 384 && cols !== 700)) return null;
  const kernel = compileKernel(match[1], `tile4-${cols}-${match[2]}`,
    ["input", "matrix", "output", "start", "end"], sourceFor(cols, Number(match[2])));
  generated.set(key, kernel);
  return kernel;
}

export function getKernelRunner(mode, matrix) {
  if (mode === "scalar") return scalarRows;
  if (mode === "tile4") return tiledRows;
  if (mode === "unrolled") return unrolledRows;
  const kernel = generatedFor(mode, matrix.cols);
  if (!kernel) throw new Error(`Generated kernel ${mode} unavailable`);
  return kernel;
}

export function runKernelRows(mode, input, matrix, output, start, end) {
  getKernelRunner(mode, matrix)(input, matrix, output, start, end);
}

function* verify(mode, matrix, rows, deadline) {
  let work = 0;
  for (const variant of [0, 1]) {
    const input = Float32Array.from({ length: matrix.cols }, (_, i) =>
      variant ? Math.cos(i * 0.31) * 0.6 - Math.sin(i * 0.07) : Math.sin(i * 0.17));
    const reference = new Float32Array(rows), output = new Float32Array(rows);
    output.fill(NaN);
    const first = variant ? 1 : 0, end = variant ? rows - 1 : rows;
    for (let row = first; row < end; row += 32) {
      if (clock() >= deadline) return null;
      const stop = Math.min(end, row + 32);
      scalarRows(input, matrix, reference, row, stop);
      runKernelRows(mode, input, matrix, output, row, stop);
      for (let i = row; i < stop; i++) if (!withinTolerance(reference[i], output[i])) return false;
      work += (stop - row) * matrix.cols * 2;
      if (work >= 100000) { work = 0; yield; }
    }
    if (variant && (!Number.isNaN(output[0]) || !Number.isNaN(output[rows - 1]))) return false;
  }
  return true;
}

function* measure(mode, input, matrix, output, rows, full, deadline) {
  const kernel = getKernelRunner(mode, matrix);
  const samples = full ? 3 : 1;
  const times = [];
  let work = 0, totalPasses = 0;
  for (let sample = 0; sample < samples && clock() < deadline; sample++) {
    let elapsed = 0, passes = 0;
    while (passes < 64 && (passes < 2 || elapsed < (full ? 4 : 2)) && clock() < deadline) {
      let passMs = 0, complete = true;
      for (let row = 0; row < rows; row += 32) {
        if (clock() >= deadline) { complete = false; break; }
        const end = Math.min(rows, row + 32), began = clock();
        kernel(input, matrix, output, row, end);
        passMs += clock() - began;
        work += (end - row) * matrix.cols;
        if (work >= 100000) { work = 0; yield; }
      }
      if (!complete) break;
      elapsed += passMs;
      passes++;
      totalPasses++;
    }
    if (passes >= 2 && elapsed >= 1) times.push(elapsed / passes);
  }
  times.sort((a, b) => a - b);
  return { ms: times.length ? times[Math.floor(times.length / 2)] : null,
    passes: totalPasses, samples: times.length, complete: times.length === samples };
}

function* benchmark(tensors, options, full) {
  const started = clock(), deadline = started + (options.maxMs ?? (full ? 60000 : 8000));
  const results = {}, verified = {};
  for (let index = 0; index < SHAPES.length; index++) {
    const name = SHAPES[index];
    const shapeDeadline = Math.min(deadline, started + (deadline - started) * 0.75 * (index + 1) / SHAPES.length);
    const matrix = tensors[name], shape = shapeOf(matrix);
    const rows = Math.min(full ? 256 : 32, matrix.rows);
    const input = Float32Array.from({ length: matrix.cols }, (_, i) => Math.sin(i * 0.17));
    const output = new Float32Array(rows), times = {}, valid = { scalar: true }, errors = {};
    for (const mode of MODES) {
      if (clock() >= shapeDeadline) break;
      if (mode.startsWith("function") || mode.startsWith("eval")) {
        if (!generatedFor(mode, matrix.cols)) { valid[mode] = false; errors[mode] = "Compiler unavailable"; continue; }
      }
      try {
        const checked = mode === "scalar" ? true : yield* verify(mode, matrix, rows, shapeDeadline);
        valid[mode] = checked;
        if (checked) times[mode] = yield* measure(mode, input, matrix, output, rows, full, shapeDeadline);
      } catch (error) {
        valid[mode] = false;
        errors[mode] = String(error).slice(0, 240);
      }
      yield;
    }
    if (times.scalar?.complete && times.scalar.ms > 0) {
      let winner = "scalar", best = times.scalar.ms;
      for (const mode of MODES.slice(1)) {
        const timing = times[mode];
        if (valid[mode] && timing?.complete && timing.ms > 0 && timing.ms < best * 0.95) {
          winner = mode;
          best = timing.ms;
        }
      }
      selected[shape] = winner;
    } else if (valid[getKernelMode(matrix)] === false) {
      selected[shape] = "scalar";
    }
    verified[shape] = valid;
    const entry = { rows, matrixRows: matrix.rows, selected: getKernelMode(matrix), times, errors,
      budgetExhausted: clock() >= shapeDeadline };
    for (const mode of MODES) entry[`${mode}Ms`] = times[mode]?.ms ?? null;
    results[shape] = entry;
  }
  const math = clock() < deadline ? yield* calibrateMathKernels({ deadline, full }) : getMathStatus().lastReport;
  lastReport = { elapsedMs: clock() - started, selected: { ...selected }, results, verified, math,
    dynamic: getDynamicStatus(), budgetExhausted: clock() >= deadline };
  return lastReport;
}
export function* calibrateKernels(tensors, options = {}) { return yield* benchmark(tensors, options, false); }
export function* benchKernels(tensors, options = {}) { return yield* benchmark(tensors, options, true); }
