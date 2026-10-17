const engines = {
  function: { available: null, error: null },
  eval: { available: null, error: null },
};
const compiled = new Map();
const failures = new Map();
let compileMs = 0;
let sourceBytes = 0;
const clock = () => typeof globalThis.performance?.now === "function" ? performance.now() : Date.now();

function construct(engine, parameters, body) {
  if (engine === "function") return new Function(...parameters, '"use strict";\n' + body);
  if (engine === "eval") return (0, eval)(`(function(${parameters.join(",")}){"use strict";\n${body}\n})`);
  throw new Error("Unknown dynamic compiler");
}

function available(engine) {
  const state = engines[engine];
  if (!state) return false;
  if (state.available === null) {
    const started = clock();
    try {
      state.available = construct(engine, ["value"], "return value + 1;")(1) === 2;
      if (!state.available) state.error = "Dynamic compiler probe returned an incorrect result";
    } catch (error) {
      state.available = false;
      state.error = String(error).slice(0, 240);
    }
    compileMs += clock() - started;
  }
  return state.available;
}

export function compileKernel(engine, name, parameters, body) {
  if (!available(engine)) return null;
  const key = `${engine}:${name}`;
  if (compiled.has(key)) return compiled.get(key);
  if (failures.has(key)) return null;
  if (!parameters.every(value => /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(value)))
    throw new Error("Invalid generated kernel parameter");
  const started = clock();
  try {
    const kernel = construct(engine, parameters, body);
    if (typeof kernel !== "function") throw new Error("Dynamic compiler did not return a function");
    compiled.set(key, kernel);
    sourceBytes += body.length;
    return kernel;
  } catch (error) {
    failures.set(key, String(error).slice(0, 240));
    return null;
  } finally {
    compileMs += clock() - started;
  }
}

export function getDynamicStatus() {
  return {
    engines: { function: { ...engines.function }, eval: { ...engines.eval } },
    compiledKernels: compiled.size,
    sourceBytes,
    compileMs,
    failures: Object.fromEntries(failures),
  };
}
