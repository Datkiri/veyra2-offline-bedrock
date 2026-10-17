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
    console.warn("[Veyra2] Không gửi được câu trả lời: " + error);
  }
}

function kernelSummary() {
  const selected = getKernelStatus().selected;
  return Object.entries(selected).map(([shape, mode]) => `${shape}:${mode}`).join(" ") || "scalar";
}

function* loadWeights() {
  const started = now();
  try {
    tensors = yield* loadModel({ workPerYield: CONFIG.workPerYield });
    console.warn(`[Veyra2] Đã nạp model sau ${(now() - started).toFixed(0)}ms; trọng số INT8=${modelBytes()}`);
    console.warn(`[Veyra2] Cache tiền tố ${tensors.prefixSeed?.ids.length || 0} token; ${(tensors.prefixSeed?.bytes || 0) / 1024} KiB`);
    try {
      const report = yield* calibrateKernels(tensors, { maxMs: CONFIG.kernelCalibrationMs });
      console.warn(`[Veyra2] Chọn kernel ${report.elapsedMs.toFixed(0)}ms; ${kernelSummary()}`);
    } catch (error) {
      console.warn(`[Veyra2] Không chọn được kernel; dùng scalar: ${error}`);
    }
  } catch (error) {
    loadError = String(error);
    console.warn("[Veyra2] Nạp model thất bại: " + loadError);
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
    tell(player, `Đo kernel ${report.elapsedMs.toFixed(0)}ms / ${ticks} ticks · ${kernelSummary()}`);
    console.warn(`[Veyra2] Kết quả đo kernel: ${JSON.stringify({ ...report, ticks })}`);
  } catch (error) {
    console.warn(`[Veyra2] Đo kernel thất bại: ${error}`);
    tell(player, "Đo kernel failed. Check the Content Log.");
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
  if (!tensors) return "chưa nạp";
  let bytes = 0;
  for (const value of Object.values(tensors)) {
    bytes += value.kind === "q8" ? value.values.byteLength + value.scales.byteLength : value.byteLength;
  }
  return `${(bytes / 1048576).toFixed(1)} MiB trọng số đã giải mã`;
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
    else tell(request.player, "Lần này model chưa tạo được câu trả lời.");
    if (keepContext && contextEnabled(request.player.id) && !resetWhileBusy.has(request.player.id)) {
      prior.push({ user: request.message, bot: text });
      history.set(request.player.id, prior.slice(-CONFIG.historyTurns));
    }
    lastStats = { historyEnabled: keepContext, ...result.stats, wallMs: now() - started, ticks: system.currentTick - startedTick, player: request.player.name };
    const stats = lastStats;
    console.warn(`[Veyra2] ${stats.promptTokens} token vào (${stats.reusedTokens} đã cache, ${stats.seededTokens} từ tiền tố, ${stats.computedPromptTokens} đã tính) + ${stats.generatedTokens} token ra; token đầu=${stats.firstTokenMs?.toFixed(0) ?? "n/a"}ms đọc=${stats.prefillMs.toFixed(0)}ms sinh=${stats.decodeMs.toFixed(0)}ms tổng=${stats.totalMs.toFixed(0)}ms; ticks=${stats.ticks} yields=${stats.yields}; skipped-final=${stats.skippedFinalBlocks}; kernels=${kernelSummary()}`);
  } catch (error) {
    sessions.delete(request.player?.id);
    console.warn("[Veyra2] Sinh câu trả lời thất bại: " + error);
    tell(request.player, "Chưa thể trả lời. Xem nhật ký nội dung để biết chi tiết.");
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
  if (loadError) { tell(request.player, "Không nạp được model. Xem nhật ký nội dung."); system.run(startNext); return; }
  if (!tensors) {
    loading = true;
    queue.unshift(request);
    system.runJob(loadWeights());
    return;
  }
  if (request.message === "__load__") {
    tell(request.player, `Đã nạp Veyra2 (${modelBytes()}).`);
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
  const normalized = input.normalize("NFD").replace(/\p{M}/gu, "").replace(/[đĐ]/g, "d").toLowerCase();
  const aliases = new Map([["trogiup", "help"], ["tro giup", "help"], ["tai", "load"],
    ["trangthai", "status"], ["trang thai", "status"], ["thongke", "stats"],
    ["thong ke", "stats"], ["huy", "cancel"], ["xoa", "reset"], ["do kernel", "bench kernels"]]);
  const lower = aliases.get(normalized) || input.toLowerCase();
  const contextCommand = /^(?:context|ngucanh|ngu canh)(?:\s+(on|off|status|bat|tat|trang thai))?$/.exec(normalized);
  if (contextCommand) {
    const value = contextCommand[1];
    if (value === "on" || value === "bat" || value === "off" || value === "tat") {
      const enabled = value === "on" || value === "bat";
      contextFlags.set(player.id, enabled);
      history.delete(player.id);
      sessions.delete(player.id);
      if (activePlayerId === player.id) resetWhileBusy.add(player.id);
      tell(player, `Đã ${enabled ? "bật" : "tắt"} ngữ cảnh và xóa lịch sử cũ. Áp dụng từ câu hỏi tiếp theo.`);
    } else {
      tell(player, `Ngữ cảnh: ${contextEnabled(player.id) ? "bật" : "tắt"}. Dùng !ai ngucanh bat hoặc !ai ngucanh tat. Mặc định: tắt.`);
    }
    return;
  }
  if (!input || lower === "help") {
    player.sendMessage("§bVeyra2:§r !ai <câu hỏi> · !ai tai · !ai trangthai · !ai thongke · !ai do kernel · !ai huy · !ai xoa");
    player.sendMessage("§7Ngữ cảnh mặc định tắt: !ai ngucanh bat | !ai ngucanh tat | !ai ngucanh. Các lệnh tiếng Anh cũ vẫn dùng được.");
    player.sendMessage("§7Veyra2-Blueberry-10M-Instruct ngoại tuyến; model trả lời chủ yếu bằng tiếng Anh; trọng số INT8.");
    return;
  }
  if (lower === "cancel") {
    if (activePlayerId !== player.id || activeJobId === null) {
      tell(player, "Bạn không có câu trả lời nào đang chạy để hủy.");
      return;
    }
    system.clearJob(activeJobId);
    activeJobId = null;
    activeProgress = null;
    sessions.delete(player.id);
    resetWhileBusy.delete(player.id);
    busy = false;
    activePlayerId = null;
    tell(player, "Đã hủy câu trả lời đang chạy.");
    system.run(startNext);
    return;
  }
  if (lower === "reset") {
    if (activePlayerId === player.id) resetWhileBusy.add(player.id);
    history.delete(player.id);
    sessions.delete(player.id);
    tell(player, "Đã xóa lịch sử hội thoại.");
    return;
  }
  if (lower === "load") {
    if (tensors) { tell(player, "Model đã được nạp."); return; }
    if (loading) { tell(player, "Đang nạp model."); return; }
    queue.push({ player, message: "__load__" });
    startNext();
    return;
  }
  if (lower === "bench kernels") {
    if (queue.length >= 4) { tell(player, "Model đang bận. Hãy thử lại sau."); return; }
    if (activePlayerId === player.id || queue.some(item => item.player.id === player.id)) { tell(player, "Yêu cầu trước của bạn vẫn đang chờ."); return; }
    queue.push({ player, message: "__bench_kernels__" });
    startNext();
    return;
  }
  if (lower === "status") {
    tell(player, `Ngữ cảnh hội thoại: ${contextEnabled(player.id) ? "bật" : "tắt"}.`);
    tell(player, `Veyra2-Blueberry-10M-Instruct · ${tensors ? "đã nạp" : loading ? "đang nạp" : "chưa nạp"}${loadError ? " · lỗi nạp model" : ""}`);
    tell(player, `INT8 tile4 · ${modelBytes()} · cache tiền tố ${tensors?.prefixSeed?.ids.length || 0} token · cửa sổ token ${CONFIG.contextLimit} · giới hạn trả lời ${CONFIG.maxReplyTokens} token · hàng đợi ${queue.length} · phiên KV ${sessions.size}`);
    if (activeProgress) {
      if (activeProgress.phase === "bench") tell(player, `Đang đo kernel; đã chạy ${((now() - activeProgress.started) / 1000).toFixed(0)}s.`);
      else tell(player, `${activeProgress.phase === "prefill" ? "Đọc" : "Trả lời"} ${activeProgress.done}/${activeProgress.total}; ${(activeProgress.counter?.macs / 1000000 || 0).toFixed(0)}M MAC; đã chạy ${((now() - activeProgress.started) / 1000).toFixed(0)}s.`);
    }
    tell(player, `Kernel ${kernelSummary()}`);
    return;
  }
  if (lower === "stats") {
    if (!lastStats) { tell(player, "Chưa có câu trả lời hoàn tất để thống kê."); tell(player, `Kernel ${kernelSummary()}`); return; }
    if (lastStats.source === "calculator") {
      tell(player, `Bộ tính JavaScript: ${lastStats.expression}; không cần suy luận model.`);
      return;
    }
    const stats = lastStats;
    tell(player, `${stats.promptTokens} token vào (${stats.reusedTokens} đã cache, ${stats.seededTokens} từ tiền tố, ${stats.computedPromptTokens} đã tính) + ${stats.generatedTokens} token ra; token đầu ${stats.firstTokenMs === null ? "n/a" : (stats.firstTokenMs / 1000).toFixed(2) + "s"}; đọc ${(stats.prefillMs / 1000).toFixed(2)}s; sinh ${(stats.decodeMs / 1000).toFixed(2)}s; tổng ${(stats.totalMs / 1000).toFixed(2)}s; ${stats.ticks} ticks; ${stats.yields} yields.`);
    tell(player, `Kernel ${kernelSummary()}`);
    return;
  }
  if (input.length > CONFIG.maxChatLength) { tell(player, `Giới hạn ${CONFIG.maxChatLength} ký tự cho mỗi câu hỏi.`); return; }
  if (queue.length >= 4) { tell(player, "Model đang bận. Hãy thử lại sau."); return; }
  if (activePlayerId === player.id || queue.some(item => item.player.id === player.id)) { tell(player, "Câu hỏi trước của bạn vẫn đang chờ."); return; }
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
      console.warn("[Veyra2] Lệnh thất bại: " + error);
      if (sender?.isValid) sender.sendMessage("§cVeyra2:§r Lệnh thất bại. Xem nhật ký nội dung.");
    }
  });
});

console.warn("[Veyra2] Addon 1.0.2 đã sẵn sàng (INT8 tile4 + cache tiền tố). Gõ !ai trogiup.");

system.runInterval(() => {
  const progress = activeProgress;
  if (!progress) return;
  const elapsed = ((now() - progress.started) / 1000).toFixed(0);
  const label = progress.phase === "prefill" ? "đọc" : progress.phase === "bench" ? "đo kernel" : "trả lời";
  const work = ((progress.counter?.macs || 0) / 1000000).toFixed(0);
  try {
    if (progress.player?.isValid) progress.player.onScreenDisplay.setActionBar(`§bAI §7${label}${progress.phase === "bench" ? "" : ` ${progress.done}/${progress.total}`} · ${elapsed}s`);
  } catch (error) {
                                                                             
  }
  if (now() - progress.lastLoggedAt >= 15000) {
    progress.lastLoggedAt = now();
    console.warn(`[Veyra2] ${label} ${progress.done}/${progress.total}; ${work}M MAC; ${elapsed}s đã chạy.`);
  }
}, 20);
