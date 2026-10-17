import { calculateReply } from "./calculator.js";
import { world, system } from "@minecraft/server";
import { CONFIG } from "./config.js";
import { loadModel } from "./model.js";
import { benchKernels, buildPrompt, calibrateKernels, DIM, generate, getKernelStatus, now } from "./inference.js";

const queue = [];
const history = new Map();
const contextFlags = new Map();
const contextEnabled = id => contextFlags.get(id) ?? CONFIG.historyEnabled;
const sessions = new Map();
const resetWhileBusy = new Set();
const MAX_CACHED_SESSIONS = 1;                                                    
let tensors = null;
let loading = false;
let busy = false;
let activePlayerId = null;
let activeJobId = null;
let activeProgress = null;
let loadError = null;
let lastStats = null;

function tell(player, text) {
  try {
    if (player?.isValid) player.sendMessage("§bAI:§r " + text);
  } catch (error) {
    console.warn("[Veyra2] Could not send chat reply: " + error);
  }
}

function kernelSummary() {
  const selected = getKernelStatus().selected;
  return Object.entries(selected).map(([shape, mode]) => `${shape}:${mode}`).join(" ") || "scalar";
}

function dynamicSummary() {
  const status = getKernelStatus();
  const label = value => value === true ? "available" : value === false ? "unavailable" : "not probed";
  return `Dynamic compilers: eval ${label(status.evalAvailable)}, Function ${label(status.functionAvailable)}; ${status.dynamic?.compiledKernels || 0} compiled kernels; compilation ${(status.dynamic?.compileMs || 0).toFixed(0)}ms.`;
}

function mathSummary() {
  return Object.entries(getKernelStatus().math?.selected || {}).map(([name, mode]) => `${name}:${mode}`).join(" ") || "builtin";
}

function* loadWeights() {
  const started = now();
  try {
    tensors = yield* loadModel({ workPerYield: CONFIG.workPerYield });
    console.warn(`[Veyra2] Model ready in ${(now() - started).toFixed(0)}ms; INT8 bytes=${modelBytes()}`);
    console.warn(`[Veyra2] Prefix snapshot ${tensors.prefixSeed?.ids.length || 0} tokens; ${(tensors.prefixSeed?.bytes || 0) / 1024} KiB`);
    try {
      const report = yield* calibrateKernels(tensors, { maxMs: CONFIG.kernelCalibrationMs });
      console.warn(`[Veyra2] Kernel calibration ${report.elapsedMs.toFixed(0)}ms; ${kernelSummary()}`);
      console.warn(`[Veyra2] ${dynamicSummary()} Math ${mathSummary()}`);
    } catch (error) {
      console.warn(`[Veyra2] Kernel calibration interrupted; keeping available kernels: ${error}`);
    }
  } catch (error) {
    loadError = String(error);
    console.warn("[Veyra2] Model load failed: " + loadError);
  } finally {
    loading = false;
    system.run(startNext);
  }
}

function* runKernelBench(player) {
  const startedTick = system.currentTick;
  try {
    const report = yield* benchKernels(tensors, { maxMs: CONFIG.kernelBenchmarkMs });
    const ticks = system.currentTick - startedTick;
    tell(player, `Kernel benchmark ${report.elapsedMs.toFixed(0)}ms / ${ticks} ticks · ${kernelSummary()}`);
    tell(player, dynamicSummary());
    tell(player, `Math kernels ${mathSummary()}`);
    if (report.budgetExhausted || Object.values(report.results || {}).some(value => value.budgetExhausted))
      tell(player, "Some measurements reached the time budget. Completed results were kept.");
    console.warn(`[Veyra2] Kernel benchmark: ${JSON.stringify({ ...report, ticks })}`);
  } catch (error) {
    console.warn(`[Veyra2] Kernel benchmark failed: ${error}`);
    tell(player, "Kernel benchmark failed. Check the Content Log.");
  } finally {
    resetWhileBusy.delete(player?.id);
    activeProgress = null;
    activeJobId = null;
    busy = false;
    activePlayerId = null;
    system.run(startNext);
  }
}

function modelBytes() {
  if (!tensors) return "unavailable";
  let bytes = 0;
  for (const value of Object.values(tensors)) {
    bytes += value.kind === "q8" ? value.values.byteLength + value.scales.byteLength : value.byteLength;
  }
  return `${(bytes / 1048576).toFixed(1)} MiB decoded weights`;
}

function* answer(request) {
  const started = now();
  const startedTick = system.currentTick;
  try {
    if (!request.player?.isValid) return;
    const keepContext = contextEnabled(request.player.id);
    const prior = keepContext ? history.get(request.player.id) || [] : [];
    const prompt = buildPrompt(request.message, prior, { maxPromptTokens: CONFIG.maxPromptTokens });
    const previousSession = keepContext ? sessions.get(request.player.id) : undefined;
                                                                             
                                                                             
    sessions.clear();
    activeProgress = { player: request.player, phase: "prefill", done: 0, total: 0, started: now(), lastLoggedAt: now() };
    const result = yield* generate(tensors, prompt, {
      maxNewTokens: CONFIG.maxReplyTokens,
      temperature: CONFIG.temperature,
      topK: CONFIG.topK,
      repetitionPenalty: CONFIG.repetitionPenalty,
      penaltyWindow: CONFIG.penaltyWindow,
      contextLimit: CONFIG.contextLimit,
      macPerYield: CONFIG.macPerYield,
      sliceMs: CONFIG.sliceMs,
      session: previousSession,
      onProgress(phase, done, total, counter) {
        if (activeProgress?.player.id === request.player.id) {
          activeProgress.phase = phase;
          activeProgress.done = done;
          activeProgress.total = total;
          activeProgress.counter = counter;
        }
      },
    });
    if (keepContext && contextEnabled(request.player.id) && !resetWhileBusy.has(request.player.id)) {
      sessions.set(request.player.id, result.session);
      while (sessions.size > MAX_CACHED_SESSIONS) sessions.delete(sessions.keys().next().value);
    }
    const text = String(result.text || "").trim();
    if (text) tell(request.player, text);
    else tell(request.player, "I could not form an answer this time.");
    if (keepContext && contextEnabled(request.player.id) && !resetWhileBusy.has(request.player.id)) {
      prior.push({ user: request.message, bot: text });
      history.set(request.player.id, prior.slice(-CONFIG.historyTurns));
    }
    lastStats = { historyEnabled: keepContext, ...result.stats, wallMs: now() - started, ticks: system.currentTick - startedTick, player: request.player.name };
    const stats = lastStats;
    console.warn(`[Veyra2] ${stats.promptTokens} prompt (${stats.reusedTokens} cached, ${stats.seededTokens} snapshot, ${stats.computedPromptTokens} computed) + ${stats.generatedTokens} generated; first=${stats.firstTokenMs?.toFixed(0) ?? "n/a"}ms prefill=${stats.prefillMs.toFixed(0)}ms decode=${stats.decodeMs.toFixed(0)}ms total=${stats.totalMs.toFixed(0)}ms; ticks=${stats.ticks} yields=${stats.yields}; skipped-final=${stats.skippedFinalBlocks}; kernels=${kernelSummary()}`);
  } catch (error) {
    sessions.delete(request.player?.id);
    console.warn("[Veyra2] Generation failed: " + error);
    tell(request.player, "I could not answer right now. Check the Content Log for details.");
  } finally {
    resetWhileBusy.delete(request.player?.id);
    activeProgress = null;
    activeJobId = null;
    busy = false;
    activePlayerId = null;
    system.run(startNext);
  }
}

function startNext() {
  if (busy || loading || !queue.length) return;
  const request = queue.shift();
  if (!request.player?.isValid) { system.run(startNext); return; }
  if (loadError) { tell(request.player, "Model loading failed. Check the Content Log."); system.run(startNext); return; }
  if (!tensors) {
    loading = true;
    queue.unshift(request);
    system.runJob(loadWeights());
    return;
  }
  if (request.message === "__load__") {
    tell(request.player, `Veyra2 loaded (${modelBytes()}).`);
    system.run(startNext);
    return;
  }
  if (request.message === "__bench_kernels__") {
    busy = true;
    activePlayerId = request.player.id;
    activeProgress = { player: request.player, phase: "bench", done: 0, total: 0, started: now(), lastLoggedAt: now() };
    activeJobId = system.runJob(runKernelBench(request.player));
    return;
  }
  busy = true;
  activePlayerId = request.player.id;
  activeJobId = system.runJob(answer(request));
}

function command(player, body) {
  const input = body.trim();
  const lower = input.toLowerCase();
  const contextCommand = /^context(?:\s+(on|off|status))?$/.exec(lower);
  if (contextCommand) {
    const value = contextCommand[1];
    if (value === "on" || value === "off") {
      const enabled = value === "on";
      contextFlags.set(player.id, enabled);
      history.delete(player.id);
      sessions.delete(player.id);
      if (activePlayerId === player.id) resetWhileBusy.add(player.id);
      tell(player, `Context is now ${enabled ? "ON" : "OFF"}. Previous history cleared. Applies to your next question.`);
    } else {
      tell(player, `Conversation context: ${contextEnabled(player.id) ? "ON" : "OFF"}. Use !ai context on or !ai context off. Default: OFF.`);
    }
    return;
  }
  if (!input || lower === "help") {
    player.sendMessage("§bVeyra2:§r !ai <message> · !ai load · !ai status · !ai stats · !ai bench kernels · !ai cancel · !ai reset");
    player.sendMessage("§7English-only Veyra2-Blueberry-10M-Instruct, offline INT8 chat. Addon by ILwHee; original model by Veyra AI.");
    player.sendMessage("§7Conversation context defaults to OFF: !ai context on | !ai context off | !ai context status.");
    player.sendMessage("§7Calculator: !ai calc (2+3)*4. Clear conversation history with !ai reset.");
    player.sendMessage("§7Use !ai bench kernels to compare built-in, eval and Function kernels on your device.");
    return;
  }
  if (lower === "cancel") {
    if (activePlayerId !== player.id || activeJobId === null) {
      tell(player, "You have no active reply to cancel.");
      return;
    }
    system.clearJob(activeJobId);
    activeJobId = null;
    activeProgress = null;
    sessions.delete(player.id);
    resetWhileBusy.delete(player.id);
    busy = false;
    activePlayerId = null;
    tell(player, "Canceled the current reply.");
    system.run(startNext);
    return;
  }
  if (lower === "reset") {
    if (activePlayerId === player.id) resetWhileBusy.add(player.id);
    history.delete(player.id);
    sessions.delete(player.id);
    tell(player, "Conversation history cleared.");
    return;
  }
  if (lower === "load") {
    if (tensors) { tell(player, "Model is already loaded."); return; }
    if (loading) { tell(player, "Model is loading."); return; }
    queue.push({ player, message: "__load__" });
    startNext();
    return;
  }
  if (lower === "bench kernels") {
    if (queue.length >= 4) { tell(player, "The model is busy. Try again in a moment."); return; }
    if (activePlayerId === player.id || queue.some(item => item.player.id === player.id)) { tell(player, "Your previous request is still queued."); return; }
    queue.push({ player, message: "__bench_kernels__" });
    startNext();
    return;
  }
  if (lower === "status") {
    tell(player, `Conversation context: ${contextEnabled(player.id) ? "ON" : "OFF"}. Default: OFF.`);
    tell(player, `Veyra2-Blueberry-10M-Instruct · ${tensors ? "loaded" : loading ? "loading" : "not loaded"}${loadError ? " · load error" : ""}`);
    tell(player, `INT8 tile4 · ${modelBytes()} · prefix snapshot ${tensors?.prefixSeed?.ids.length || 0} tokens · context ${CONFIG.contextLimit} · reply cap ${CONFIG.maxReplyTokens} tokens · queue ${queue.length} · KV sessions ${sessions.size}`);
    if (activeProgress) {
      if (activeProgress.phase === "bench") tell(player, `Kernel benchmark running; elapsed ${((now() - activeProgress.started) / 1000).toFixed(0)}s.`);
      else tell(player, `${activeProgress.phase} ${activeProgress.done}/${activeProgress.total}; ${(activeProgress.counter?.macs / 1000000 || 0).toFixed(0)}M MAC; elapsed ${((now() - activeProgress.started) / 1000).toFixed(0)}s.`);
    }
    tell(player, `Kernels ${kernelSummary()}`);
    tell(player, dynamicSummary());
    tell(player, `Math kernels ${mathSummary()}`);
    return;
  }
  if (lower === "stats") {
    if (!lastStats) { tell(player, "No completed reply to report yet."); tell(player, `Kernels ${kernelSummary()}`); return; }
    if (lastStats.source === "calculator") {
      tell(player, `JavaScript calculator: ${lastStats.expression}; no model inference.`);
      return;
    }
    const stats = lastStats;
    tell(player, `${stats.promptTokens} prompt (${stats.reusedTokens} cached, ${stats.seededTokens} snapshot, ${stats.computedPromptTokens} computed) + ${stats.generatedTokens} generated; first ${stats.firstTokenMs === null ? "n/a" : (stats.firstTokenMs / 1000).toFixed(2) + "s"}; prefill ${(stats.prefillMs / 1000).toFixed(2)}s; decode ${(stats.decodeMs / 1000).toFixed(2)}s; total ${(stats.totalMs / 1000).toFixed(2)}s; ${stats.ticks} ticks; ${stats.yields} yields.`);
    tell(player, `Kernels ${kernelSummary()}`);
    return;
  }
  if (input.length > CONFIG.maxChatLength) { tell(player, `Keep messages under ${CONFIG.maxChatLength} characters.`); return; }
  if (queue.length >= 4) { tell(player, "The model is busy. Try again in a moment."); return; }
  if (activePlayerId === player.id || queue.some(item => item.player.id === player.id)) { tell(player, "Your previous message is still queued."); return; }
  const math = calculateReply(input);
  if (math) {
    tell(player, math.text);
    if (contextEnabled(player.id)) {
      const prior = history.get(player.id) || [];
      prior.push({ user: input, bot: math.text });
      history.set(player.id, prior.slice(-CONFIG.historyTurns));
    }
    lastStats = { source: "calculator", expression: math.expression };
    return;
  }
  queue.push({ player, message: input });
  startNext();
}

world.beforeEvents.chatSend.subscribe(event => {
  if (!/^!ai(?:\s|$)/i.test(event.message)) return;
  event.cancel = true;
  const sender = event.sender;
  const body = event.message.slice(3);
  system.run(() => {
    try { if (sender?.isValid) command(sender, body); }
    catch (error) {
      console.warn("[Veyra2] Command failed: " + error);
      if (sender?.isValid) sender.sendMessage("§cVeyra2:§r Command failed. Check the Content Log.");
    }
  });
});

console.warn("[Veyra2] English-only chat pack 1.1.0 ready (INT8 + eval/Function kernels + prefix snapshot). Type !ai help.");

system.runInterval(() => {
  const progress = activeProgress;
  if (!progress) return;
  const elapsed = ((now() - progress.started) / 1000).toFixed(0);
  const label = progress.phase === "prefill" ? "reading" : progress.phase === "bench" ? "bench" : "reply";
  const work = ((progress.counter?.macs || 0) / 1000000).toFixed(0);
  try {
    if (progress.player?.isValid) progress.player.onScreenDisplay.setActionBar(`§bAI §7${label}${progress.phase === "bench" ? "" : ` ${progress.done}/${progress.total}`} · ${elapsed}s`);
  } catch (error) {
                                                                                
  }
  if (now() - progress.lastLoggedAt >= 15000) {
    progress.lastLoggedAt = now();
    console.warn(`[Veyra2] ${label} ${progress.done}/${progress.total}; ${work}M MAC; ${elapsed}s elapsed.`);
  }
}, 20);
