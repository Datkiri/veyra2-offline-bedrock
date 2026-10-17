                                                                                  
export function calculateReply(message) {
  let expression = String(message).trim().replace(/×/g, "*").replace(/÷/g, "/").replace(/−/g, "-");
  expression = expression.replace(/^(?:what(?:'s|\s+is)\s+the\s+result\s+of|what(?:'s|\s+is)|calculate|compute|solve|tính|tinh|kết quả của|ket qua cua)\s+/i, "");
  expression = expression.replace(/\s*(?:=\s*\??|\?)\s*$/, "").trim();
  if (!expression || expression.length > 128 || !/^[\d.\s()+*/-]+$/.test(expression) || !/[+*/-]/.test(expression)) return null;
  const tokens = expression.match(/\d+(?:\.\d*)?|\.\d+|[()+*/-]/g) || [];
  if (tokens.join("") !== expression.replace(/\s/g, "")) return null;
  let at = 0, depth = 0;
  function bounded(value) {
    if (!Number.isFinite(value) || Math.abs(value) > 1e12) throw new Error("Kết quả vượt phạm vi hỗ trợ");
    return value;
  }
  function atom() {
    if (++depth > 16) throw new Error("Biểu thức lồng quá sâu");
    let value;
    const token = tokens[at++];
    if (token === "+" || token === "-") value = (token === "-" ? -1 : 1) * atom();
    else if (token === "(") {
      value = sum();
      if (tokens[at++] !== ")") throw new Error("Dấu ngoặc chưa cân bằng");
    } else {
      if (!token || !/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(token)) throw new Error("Cần một số");
      value = Number(token);
    }
    depth--;
    return bounded(value);
  }
  function product() {
    let value = atom();
    while (tokens[at] === "*" || tokens[at] === "/") {
      const op = tokens[at++], rhs = atom();
      if (op === "/" && rhs === 0) throw new Error("Không thể chia cho 0");
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
    if (error.message === "Không thể chia cho 0" || error.message === "Kết quả vượt phạm vi hỗ trợ")
      return { expression, text: error.message + "." };
    return null;
  }
}
