                                                                                  
export function calculateReply(message) {
  let expression = String(message).trim().replace(/×/g, "*").replace(/÷/g, "/").replace(/−/g, "-");
  expression = expression.replace(/^(?:what(?:'s|\s+is)\s+the\s+result\s+of|what(?:'s|\s+is)|calculate|calc|compute|solve)\s+/i, "");
  expression = expression.replace(/\s*(?:=\s*\??|\?)\s*$/, "").trim();
  if (!expression || expression.length > 128 || !/^[\d.\s()+*/-]+$/.test(expression) || !/[+*/-]/.test(expression)) return null;
  const tokens = expression.match(/\d+(?:\.\d*)?|\.\d+|[()+*/-]/g) || [];
  if (tokens.join("") !== expression.replace(/\s/g, "")) return null;
  let at = 0, depth = 0;
  function bounded(value) {
    if (!Number.isFinite(value) || Math.abs(value) > 1e12) throw new Error("Result outside supported range");
    return value;
  }
  function atom() {
    if (++depth > 16) throw new Error("Expression too deeply nested");
    let value;
    const token = tokens[at++];
    if (token === "+" || token === "-") value = (token === "-" ? -1 : 1) * atom();
    else if (token === "(") {
      value = sum();
      if (tokens[at++] !== ")") throw new Error("Unbalanced parentheses");
    } else {
      if (!token || !/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(token)) throw new Error("Expected a number");
      value = Number(token);
    }
    depth--;
    return bounded(value);
  }
  function product() {
    let value = atom();
    while (tokens[at] === "*" || tokens[at] === "/") {
      const op = tokens[at++], rhs = atom();
      if (op === "/" && rhs === 0) throw new Error("Cannot divide by zero");
      value = bounded(op === "*" ? value * rhs : value / rhs);
    }
    return value;
  }
  function sum() {
    let value = product();
    while (tokens[at] === "+" || tokens[at] === "-") {
      const op = tokens[at++], rhs = product();
      value = bounded(op === "+" ? value + rhs : value - rhs);
    }
    return value;
  }
  try {
    const value = sum();
    if (at !== tokens.length) return null;
    return { expression, text: String(Number(value.toPrecision(12))) };
  } catch (error) {
    if (error.message === "Cannot divide by zero" || error.message === "Result outside supported range")
      return { expression, text: error.message + "." };
    return null;
  }
}
