import { DIM } from "./generated/model_config.js";
import { compileKernel } from "./dynamic.js";

const H = DIM.hidden, HD = DIM.headSize, F = DIM.ffn;
const clock = () => typeof globalThis.performance?.now === "function" ? performance.now() : Date.now();
const selected = Object.create(null);
let lastReport = null;

const builtin = Object.freeze({
  prepareRope(position, cos, sin) {
    for (let i = 0; i < HD / 2; i++) {
      const angle = position * Math.pow(DIM.ropeTheta, -(2 * i) / HD);
      cos[i] = Math.cos(angle);
      sin[i] = Math.sin(angle);
    }
  },
  embedding(matrix, row, output) {
    const stride = matrix.layout === "tile4" ? 4 : 1;
    const base = stride === 4 ? (row - (row & 3)) * matrix.cols + (row & 3) : row * matrix.cols;
    const scale = matrix.scales[row];
    for (let i = 0; i < matrix.cols; i++) output[i] = matrix.values[base + i * stride] * scale;
  },
  norm(input, weight, output) {
    let sum = 0;
    for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
    const inverse = 1 / Math.sqrt(sum / input.length + DIM.rmsEpsilon);
    for (let i = 0; i < input.length; i++) output[i] = input[i] * inverse * weight[i];
  },
  headNorm(vector, heads, weight) {
    for (let head = 0; head < heads; head++) {
      const base = head * HD;
      let sum = 0;
      for (let i = 0; i < HD; i++) sum += vector[base + i] * vector[base + i];
      const inverse = 1 / Math.sqrt(sum / HD + DIM.rmsEpsilon);
      for (let i = 0; i < HD; i++) vector[base + i] *= inverse * weight[i];
    }
  },
  rope(vector, heads, cos, sin) {
    for (let head = 0; head < heads; head++) {
      const base = head * HD;
      for (let i = 0; i < HD / 2; i++) {
        const first = vector[base + i], second = vector[base + i + HD / 2];
        vector[base + i] = first * cos[i] - second * sin[i];
        vector[base + i + HD / 2] = second * cos[i] + first * sin[i];
      }
    }
  },
  dot(query, qBase, keys, cacheBase) {
    let sum = 0;
    for (let i = 0; i < HD; i++) sum += query[qBase + i] * keys[cacheBase + i];
    return sum;
  },
  weightedAdd(output, outBase, values, cacheBase, probability) {
    for (let i = 0; i < HD; i++) output[outBase + i] += probability * values[cacheBase + i];
  },
  swiglu(gate, up) {
    for (let i = 0; i < F; i++) {
      const value = gate[i];
      gate[i] = (value / (1 + Math.exp(-value))) * up[i];
    }
  },
  residual(first, second, output) {
    for (let i = 0; i < H; i++) output[i] = first[i] + second[i];
  },
});

export const mathKernels = { ...builtin };
export const getMathStatus = () => ({ selected: { ...selected }, lastReport });

function unroll(length, statement) {
  const limit = length - length % 8;
  let result = `for(let i=0;i<${limit};i+=8){`;
  for (let j = 0; j < 8; j++) result += statement(j ? `(i+${j})` : "i", j);
  result += "}";
  if (limit !== length) result += `for(let i=${limit};i<${length};i++){${statement("i", 0)}}`;
  return result;
}

function source() {
  const normSum = unroll(H, i => `sum+=input[${i}]*input[${i}];`);
  const normWrite = unroll(H, i => `output[${i}]=input[${i}]*inverse*weight[${i}];`);
  const headSum = unroll(HD, i => `sum+=vector[base+${i}]*vector[base+${i}];`);
  const headWrite = unroll(HD, i => `vector[base+${i}]*=inverse*weight[${i}];`);
  let angles = "";
  for (let i = 0; i < HD / 2; i++)
    angles += `const angle${i}=position*${Math.pow(DIM.ropeTheta, -(2 * i) / HD)};cos[${i}]=Math.cos(angle${i});sin[${i}]=Math.sin(angle${i});`;
  return `return {
    prepareRope:function(position,cos,sin){${angles}},
    embedding:function(matrix,row,output){
      if(matrix.cols!==${H}||matrix.layout!=="tile4")return fallback.embedding(matrix,row,output);
      const values=matrix.values,base=(row-(row&3))*${H}+(row&3),scale=matrix.scales[row];
      ${unroll(H, i => `output[${i}]=values[base+${i}*4]*scale;`)}
    },
    norm:function(input,weight,output){
      if(input.length!==${H})return fallback.norm(input,weight,output);
      let sum=0;${normSum}
      const inverse=1/Math.sqrt(sum/${H}+${DIM.rmsEpsilon});${normWrite}
    },
    headNorm:function(vector,heads,weight){
      for(let head=0;head<heads;head++){
        const base=head*${HD};let sum=0;${headSum}
        const inverse=1/Math.sqrt(sum/${HD}+${DIM.rmsEpsilon});${headWrite}
      }
    },
    rope:function(vector,heads,cos,sin){
      for(let head=0;head<heads;head++){
        const base=head*${HD};
        ${unroll(HD / 2, (i, j) => `const a${j}=vector[base+${i}],b${j}=vector[base+${i}+${HD / 2}];vector[base+${i}]=a${j}*cos[${i}]-b${j}*sin[${i}];vector[base+${i}+${HD / 2}]=b${j}*cos[${i}]+a${j}*sin[${i}];`)}
      }
    },
    dot:function(query,qBase,keys,cacheBase){
      let sum=0;${unroll(HD, i => `sum+=query[qBase+${i}]*keys[cacheBase+${i}];`)}return sum;
    },
    weightedAdd:function(output,outBase,values,cacheBase,probability){
      ${unroll(HD, i => `output[outBase+${i}]+=probability*values[cacheBase+${i}];`)}
    },
    swiglu:function(gate,up){
      ${unroll(F, (i, j) => `const g${j}=gate[${i}];gate[${i}]=(g${j}/(1+Math.exp(-g${j})))*up[${i}];`)}
    },
    residual:function(first,second,output){
      ${unroll(H, i => `output[${i}]=first[${i}]+second[${i}];`)}
    }
  };`;
}

function taskFor(name, variant) {
  const vector = (length, phase = 0) => Float32Array.from({ length }, (_, i) => Math.sin(i * 0.13 + phase) * 0.7);
  const weight = length => Float32Array.from({ length }, (_, i) => 0.8 + Math.cos(i * 0.17) * 0.2);
  let args, output, value;
  if (name === "prepareRope") {
    output = new Float32Array(HD);
    args = [variant ? 127 : 17, output.subarray(0, HD / 2), output.subarray(HD / 2)];
  } else if (name === "embedding") {
    output = new Float32Array(H);
    args = [{ cols: H, rows: 8, layout: "tile4", values: Int8Array.from({ length: H * 8 }, (_, i) => i % 255 - 127),
      scales: Float32Array.from({ length: 8 }, (_, i) => (i + 1) / 200) }, variant ? 7 : 1, output];
  } else if (name === "norm") {
    const input = vector(H, variant);
    output = variant ? input : new Float32Array(H);
    args = [input, weight(H), output];
  } else if (name === "headNorm" || name === "rope") {
    const heads = variant ? DIM.kvHeads : DIM.heads;
    output = vector(heads * HD, variant);
    args = name === "headNorm" ? [output, heads, weight(HD)] : [output, heads,
      Float32Array.from({ length: HD / 2 }, (_, i) => Math.cos(i * 0.11)),
      Float32Array.from({ length: HD / 2 }, (_, i) => Math.sin(i * 0.11))];
  } else if (name === "dot") {
    args = [vector(HD + 8, variant), 2, vector(HD + 8, 0.4 + variant), 3];
  } else if (name === "weightedAdd") {
    output = vector(H, variant);
    args = [output, 3, vector(HD + 8, 0.4), 5, variant ? -0.2 : 0.37];
  } else if (name === "swiglu") {
    output = vector(F, variant);
    args = [output, vector(F, 0.4)];
  } else {
    const input = vector(H, variant);
    output = variant ? input : new Float32Array(H);
    args = [input, vector(H, 0.4), output];
  }
  return {
    run(fn) { value = fn(...args); },
    read() { return output ? output.slice() : [value]; },
  };
}

function matches(first, second) {
  return first.length === second.length && first.every((value, i) => Number.isFinite(value) && Number.isFinite(second[i]) &&
    Math.abs(value - second[i]) <= 1e-6 + 1e-6 * Math.max(Math.abs(value), Math.abs(second[i])));
}

function measure(name, kernel, full) {
  const cost = { prepareRope: HD * 4, embedding: H, norm: H * 3, headNorm: H * 3, rope: H * 3,
    dot: HD, weightedAdd: HD, swiglu: F * 4, residual: H }[name];
  const calls = Math.max(1, Math.floor(32768 / cost));
  const times = [];
  for (let sample = 0; sample < (full ? 3 : 1); sample++) {
    const task = taskFor(name, 0);
    const started = clock();
    for (let repeat = 0; repeat < calls; repeat++) task.run(kernel);
    times.push((clock() - started) / calls);
  }
  times.sort((a, b) => a - b);
  return { ms: times[Math.floor(times.length / 2)], calls };
}

export function* calibrateMathKernels(options = {}) {
  const started = clock(), deadline = options.deadline ?? Infinity;
  const bundles = {};
  const body = source();
  for (const engine of ["function", "eval"]) {
    const factory = compileKernel(engine, `math8-${H}-${HD}-${F}`, ["fallback"], body);
    if (factory) bundles[`${engine}8`] = factory(builtin);
    yield;
  }
  const results = {}, verified = {};
  for (const name of Object.keys(builtin)) {
    if (clock() >= deadline) break;
    const times = {}, valid = { builtin: true };
    const candidates = { builtin: builtin[name] };
    for (const [mode, bundle] of Object.entries(bundles)) {
      valid[mode] = true;
      try {
        for (const variant of [0, 1]) {
          const first = taskFor(name, variant), second = taskFor(name, variant);
          first.run(builtin[name]);
          second.run(bundle[name]);
          if (!matches(first.read(), second.read())) valid[mode] = false;
        }
      } catch (error) {
        valid[mode] = false;
      }
      if (valid[mode]) candidates[mode] = bundle[name];
    }
    let winner = "builtin", best = Infinity;
    for (const [mode, kernel] of Object.entries(candidates)) {
      if (clock() >= deadline) break;
      const timing = measure(name, kernel, options.full === true);
      times[mode] = timing;
      if (mode === "builtin") best = timing.ms;
      else if (best > 0 && timing.ms > 0 && timing.ms < best * 0.9 && timing.ms * timing.calls >= 1) {
        winner = mode;
        best = timing.ms;
      }
      yield;
    }
    if (times.builtin) {
      mathKernels[name] = candidates[winner];
      selected[name] = winner;
    }
    verified[name] = valid;
    results[name] = { selected: selected[name] || "builtin", times };
  }
  lastReport = { elapsedMs: clock() - started, selected: { ...selected }, results, verified };
  return lastReport;
}
