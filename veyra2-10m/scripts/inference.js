import { decode, encode } from "./tokenizer.js";
import { benchKernels, calibrateKernels, demoteKernel, getKernelMode,
  getKernelRunner, getKernelStatus, maxRowsPerSlice, scalarRows, withinTolerance } from "./kernels.js";
import { DIM, SYSTEM_PREFIX } from "./generated/model_config.js";
import { mathKernels } from "./math_kernels.js";
export { benchKernels, calibrateKernels, getKernelStatus, DIM };
const H = DIM.hidden;
const HD = DIM.headSize;
const KV_WIDTH = DIM.kvHeads * HD;
const VERIFY_ROW = new Float32Array(1);
const perf = globalThis.performance;
export const now = perf && typeof perf.now === "function" ? () => perf.now() : () => Date.now();

const S = {
  hidden: new Float32Array(H),
  residual: new Float32Array(H),
  residual2: new Float32Array(H),
  norm: new Float32Array(H),
  q: new Float32Array(H),
  k: new Float32Array(KV_WIDTH),
  v: new Float32Array(KV_WIDTH),
  attention: new Float32Array(H),
  projection: new Float32Array(H),
  gate: new Float32Array(DIM.ffn),
  up: new Float32Array(DIM.ffn),
  mlp: new Float32Array(H),
  scores: new Float32Array(8192),
  ropeCos: new Float32Array(HD / 2),
  ropeSin: new Float32Array(HD / 2),
  logits: new Float32Array(DIM.vocab),
  topIds: new Int32Array(64),
  topScores: new Float32Array(64),
  seen: new Uint8Array(DIM.vocab),
};

function* linearInto(input, matrix, output, counter) {
  const { rows, cols } = matrix;
  let mode = getKernelMode(matrix);
  let kernel;
  try { kernel = getKernelRunner(mode, matrix); }
  catch (error) { demoteKernel(matrix, error); mode = "scalar"; kernel = scalarRows; }
  for (let row = 0; row < rows;) {
    const remaining = Math.max(1, Math.floor((counter.budget - counter.work) / cols));
    const end = Math.min(rows, row + Math.min(maxRowsPerSlice(), remaining));
    try { kernel(input, matrix, output, row, end); }
    catch (error) {
      demoteKernel(matrix, error);
      mode = "scalar";
      kernel = scalarRows;
      scalarRows(input, matrix, output, row, end);
    }
    if (row === 0 && mode !== "scalar") {
      scalarRows(input, matrix, VERIFY_ROW, 0, 1);
      if (!withinTolerance(VERIFY_ROW[0], output[0])) {
        demoteKernel(matrix, "first-row result differed from scalar");
        mode = "scalar";
        kernel = scalarRows;
        scalarRows(input, matrix, output, row, end);
      }
    }
    counter.work += (end - row) * cols;
    counter.macs += (end - row) * cols;
    row = end;
    if (counter.work >= counter.budget || (counter.sliceMs > 0 && now() - counter.sliceStart >= counter.sliceMs)) {
      counter.work = 0;
      counter.yields++;
      yield;
      counter.sliceStart = now();
    }
  }
}

function* attendInto(query, keyCache, valueCache, position, output, counter) {
  const scale = 1 / Math.sqrt(HD);
  const dotKernel = mathKernels.dot, weightedKernel = mathKernels.weightedAdd;
  output.fill(0);
  for (let head = 0; head < DIM.heads; head++) {
    const qBase = head * HD;
    const kvBase = Math.floor(head / (DIM.heads / DIM.kvHeads)) * HD;
    let maximum = -Infinity;
    for (let p = 0; p <= position; p++) {
      const cacheBase = p * KV_WIDTH + kvBase;
      const dot = dotKernel(query, qBase, keyCache, cacheBase);
      const score = dot * scale;
      S.scores[p] = score;
      if (score > maximum) maximum = score;
      counter.work += HD;
      counter.macs += HD;
      if (counter.work >= counter.budget || (counter.sliceMs > 0 && now() - counter.sliceStart >= counter.sliceMs)) {
        counter.work = 0;
        counter.yields++;
        yield;
        counter.sliceStart = now();
      }
    }
    let total = 0;
    for (let p = 0; p <= position; p++) {
      const value = Math.exp(S.scores[p] - maximum);
      S.scores[p] = value;
      total += value;
    }
    const inverseTotal = total > 0 ? 1 / total : 0;
    for (let p = 0; p <= position; p++) {
      const probability = S.scores[p] * inverseTotal;
      const cacheBase = p * KV_WIDTH + kvBase;
      weightedKernel(output, qBase, valueCache, cacheBase, probability);
      counter.work += HD;
      counter.macs += HD;
      if (counter.work >= counter.budget || (counter.sliceMs > 0 && now() - counter.sliceStart >= counter.sliceMs)) {
        counter.work = 0;
        counter.yields++;
        yield;
        counter.sliceStart = now();
      }
    }
  }
}

function layerTensor(tensors, layer, suffix) {
  return tensors[`model.layers.${layer}.${suffix}`];
}

function* forwardToken(tensors, tokenId, position, cache, counter, needHidden = true) {
  const embedding = tensors["model.embed_tokens.weight"];
  if (tokenId < 0 || tokenId >= DIM.vocab) throw new Error(`Token id outside Veyra2 vocab: ${tokenId}`);
  if (position >= cache.limit) throw new Error(`Context limit ${cache.limit} exceeded`);
  mathKernels.embedding(embedding, tokenId, S.hidden);
  mathKernels.prepareRope(position, S.ropeCos, S.ropeSin);

  for (let layer = 0; layer < DIM.layers; layer++) {
    const key = cache.keys[layer];
    const value = cache.values[layer];
    const inputNorm = tensors[`model.layers.${layer}.input_layernorm.weight`];
    const postNorm = tensors[`model.layers.${layer}.post_attention_layernorm.weight`];
    const base = position * KV_WIDTH;

    S.residual.set(S.hidden);
    mathKernels.norm(S.hidden, inputNorm, S.norm);
    const cacheOnly = !needHidden && layer === DIM.layers - 1;
    if (!cacheOnly) yield* linearInto(S.norm, layerTensor(tensors, layer, "self_attn.q_proj.weight"), S.q, counter);
    yield* linearInto(S.norm, layerTensor(tensors, layer, "self_attn.k_proj.weight"), S.k, counter);
    yield* linearInto(S.norm, layerTensor(tensors, layer, "self_attn.v_proj.weight"), S.v, counter);
    if (!cacheOnly) mathKernels.headNorm(S.q, DIM.heads, layerTensor(tensors, layer, "self_attn.q_norm.weight"));
    mathKernels.headNorm(S.k, DIM.kvHeads, layerTensor(tensors, layer, "self_attn.k_norm.weight"));
    if (!cacheOnly) mathKernels.rope(S.q, DIM.heads, S.ropeCos, S.ropeSin);
    mathKernels.rope(S.k, DIM.kvHeads, S.ropeCos, S.ropeSin);
    key.set(S.k, base);
    value.set(S.v, base);
                                                                                    
                                                                        
    if (cacheOnly) {
      counter.skippedFinalBlocks++;
      return null;
    }

    yield* attendInto(S.q, key, value, position, S.attention, counter);
    yield* linearInto(S.attention, layerTensor(tensors, layer, "self_attn.o_proj.weight"), S.projection, counter);
    mathKernels.residual(S.residual, S.projection, S.hidden);

    S.residual2.set(S.hidden);
    mathKernels.norm(S.hidden, postNorm, S.norm);
    yield* linearInto(S.norm, layerTensor(tensors, layer, "mlp.gate_proj.weight"), S.gate, counter);
    yield* linearInto(S.norm, layerTensor(tensors, layer, "mlp.up_proj.weight"), S.up, counter);
    mathKernels.swiglu(S.gate, S.up);
    yield* linearInto(S.gate, layerTensor(tensors, layer, "mlp.down_proj.weight"), S.mlp, counter);
    mathKernels.residual(S.residual2, S.mlp, S.hidden);
  }
  mathKernels.norm(S.hidden, tensors["model.norm.weight"], S.hidden);
  return S.hidden;
}

function createCache(limit) {
  if (limit > S.scores.length) throw new Error(`Context limit cannot exceed ${S.scores.length}`);
  const keys = [];
  const values = [];
  const length = limit * KV_WIDTH;
  for (let i = 0; i < DIM.layers; i++) {
    keys.push(new Float32Array(length));
    values.push(new Float32Array(length));
  }
  return { keys, values, limit };
}

function prepareSeen(ids, window, penalty) {
  S.seen.fill(0);
  if (penalty === 1) return;
  const from = Math.max(0, ids.length - window);
  for (let i = from; i < ids.length; i++) S.seen[ids[i]] = 1;
}

function* sample(logits, options, counter) {
  const temperature = options.temperature;
  const topK = Math.max(1, Math.min(options.topK, S.topIds.length));
  const penalty = options.repetitionPenalty;
  const ids = S.topIds;
  const scores = S.topScores;
  let count = 0;
  for (let id = 0; id < DIM.vocab; id++) {
    let score = logits[id];
    if (penalty !== 1 && S.seen[id]) score = score < 0 ? score * penalty : score / penalty;
    score = temperature > 0 ? score / temperature : score;
    if (count < topK) {
      let at = count++;
      while (at > 0 && scores[at - 1] < score) {
        scores[at] = scores[at - 1]; ids[at] = ids[at - 1]; at--;
      }
      scores[at] = score; ids[at] = id;
    } else if (score > scores[count - 1]) {
      let at = count - 1;
      while (at > 0 && scores[at - 1] < score) {
        scores[at] = scores[at - 1]; ids[at] = ids[at - 1]; at--;
      }
      scores[at] = score; ids[at] = id;
    }
    if ((id & 255) === 255 && (counter.work >= counter.budget || (counter.sliceMs > 0 && now() - counter.sliceStart >= counter.sliceMs))) {
      counter.work = 0;
      counter.yields++;
      yield;
      counter.sliceStart = now();
    }
  }

  if (temperature <= 0) return ids[0];
  const maximum = scores[0];
  let total = 0;
  for (let i = 0; i < count; i++) total += Math.exp(scores[i] - maximum);
  let target = Math.random() * total;
  for (let i = 0; i < count; i++) {
    target -= Math.exp(scores[i] - maximum);
    if (target <= 0) return ids[i];
  }
  return ids[count - 1];
}

export function buildPrompt(message, previousTurns = [], options = {}) {
  const maxPromptTokens = options.maxPromptTokens ?? 224;
  const system = options.systemPrompt ?? "";
  const turns = previousTurns.slice(-2).map(turn =>
    `<|im_start|>user\n${turn.user}<|im_end|>\n<|im_start|>assistant\n${turn.bot}<|im_end|>\n`
  );
  const current = `<|im_start|>user\n${message}<|im_end|>\n<|im_start|>assistant\n`;
  const prefix = SYSTEM_PREFIX + (system ? `\n\n${system}` : "") + "<|im_end|>\n";
  while (turns.length && encode(prefix + turns.join("") + current).length > maxPromptTokens) turns.shift();
  const prompt = prefix + turns.join("") + current;
  if (encode(prompt).length > maxPromptTokens) throw new Error("Message is too long for Veyra2 context");
  return prompt;
}

function* generateInner(tensors, prompt, options) {
  const started = now();
  const onProgress = typeof options.onProgress === "function" ? options.onProgress : null;
  const promptIds = encode(prompt);
  const contextLimit = options.contextLimit ?? 256;
  if (!promptIds.length || promptIds.length > contextLimit) throw new Error("Prompt exceeds Veyra2 context limit");
  let previous = options.session;
  const canReuse = previous?.cache?.limit === contextLimit && Array.isArray(previous.ids) && previous.hidden?.length === H;
  const cache = canReuse ? previous.cache : createCache(contextLimit);
  let reusedTokens = 0;
  if (canReuse) {
    const shared = Math.min(promptIds.length, previous.ids.length);
    while (reusedTokens < shared && promptIds[reusedTokens] === previous.ids[reusedTokens]) reusedTokens++;
                                                                         
                                                                                 
    if (reusedTokens === promptIds.length && previous.ids.length > promptIds.length) reusedTokens--;
  }
  let seededTokens = 0;
  const seed = options.usePrefixCache === false ? null : tensors.prefixSeed;
  if (seed && seed.ids.length <= contextLimit) {
    let matched = 0;
    const shared = Math.min(promptIds.length, seed.ids.length);
    while (matched < shared && promptIds[matched] === seed.ids[matched]) matched++;
    if (matched === promptIds.length && seed.ids.length > promptIds.length) matched--;
    if (matched > reusedTokens) {
                                                                                   
      for (let layer = 0; layer < DIM.layers; layer++) {
        cache.keys[layer].set(seed.keys[layer].subarray(0, matched * KV_WIDTH));
        cache.values[layer].set(seed.values[layer].subarray(0, matched * KV_WIDTH));
        yield;
      }
      reusedTokens = seededTokens = matched;
      previous = seed;
    }
  }
  const counter = {
    budget: Math.max(4096, options.macPerYield ?? 100000),
    sliceMs: Math.max(0, options.sliceMs ?? 0),
    sliceStart: now(),
    now,
    work: 0,
    macs: 0,
    yields: 0,
    skippedFinalBlocks: 0,
  };
  const generation = {
    maxNewTokens: Math.max(1, Math.min(options.maxNewTokens ?? 48, contextLimit - promptIds.length + 1)),
    temperature: options.temperature ?? 0.7,
    topK: options.topK ?? 32,
    repetitionPenalty: options.repetitionPenalty ?? 1.08,
    penaltyWindow: options.penaltyWindow ?? 64,
  };

  const prefillStart = now();
  let hidden;
  if (onProgress) onProgress("prefill", 0, promptIds.length - reusedTokens, counter);
  if (reusedTokens === promptIds.length) {
    S.hidden.set(previous.hidden);
    hidden = S.hidden;
  }
  for (let i = reusedTokens; i < promptIds.length; i++) {
    hidden = yield* forwardToken(tensors, promptIds[i], i, cache, counter,
      i === promptIds.length - 1 || options.fullPrefill === true);
    if (onProgress) onProgress("prefill", i + 1 - reusedTokens, promptIds.length - reusedTokens, counter);
    yield;
  }
  const prefillMs = now() - prefillStart;

  const allIds = promptIds.slice();
  const processedIds = promptIds.slice();
  const outputIds = [];
  const decodeStart = now();
  let firstTokenMs = null;
  let stopReason = "cap";
  if (onProgress) onProgress("decode", 0, generation.maxNewTokens, counter);
  for (let step = 0; step < generation.maxNewTokens; step++) {
    prepareSeen(allIds, generation.penaltyWindow, generation.repetitionPenalty);
    yield* linearInto(hidden, tensors["model.embed_tokens.weight"], S.logits, counter);
    if (typeof options.onLogits === "function") options.onLogits(S.logits, step);
    const next = yield* sample(S.logits, generation, counter);
    if (firstTokenMs === null) firstTokenMs = now() - started;
    if (onProgress) onProgress("decode", step + 1, generation.maxNewTokens, counter);
    if (DIM.stopIds.includes(next)) {
      stopReason = "eos";
      break;
    }
    outputIds.push(next);
    allIds.push(next);
    if (step + 1 < generation.maxNewTokens) {
      hidden = yield* forwardToken(tensors, next, promptIds.length + step, cache, counter);
      processedIds.push(next);
      yield;
    }
  }
  const decodeMs = now() - decodeStart;
  const totalMs = now() - started;
  const stats = {
    promptTokens: promptIds.length,
    reusedTokens,
    seededTokens,
    skippedFinalBlocks: counter.skippedFinalBlocks,
    computedPromptTokens: promptIds.length - reusedTokens,
    generatedTokens: outputIds.length,
    firstTokenMs,
    prefillMs,
    decodeMs,
    totalMs,
    yields: counter.yields,
    macs: counter.macs,
    macPerSecond: totalMs > 0 ? counter.macs * 1000 / totalMs : 0,
    stopReason,
  };
  return {
    text: decode(outputIds).replace(/<\|im_(?:start|end)\|>/g, "").trim(),
    ids: outputIds,
    stats,
    session: { cache, ids: processedIds, hidden: new Float32Array(hidden) },
  };
}

export function* generate(tensors, prompt, options = {}) {
  return yield* generateInner(tensors, prompt, options);
}
