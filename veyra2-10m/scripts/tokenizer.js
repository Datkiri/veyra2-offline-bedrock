import { TOKENIZER } from "./generated/tokenizer_data.js";

const TOKEN_TO_ID = new Map(TOKENIZER.tokens.map((token, id) => [token, id]));
const MERGE_RANK = new Map(TOKENIZER.merges.map((pair, rank) => [pair[0] + " " + pair[1], rank]));
const PRETOKEN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const SPECIAL_RE = /<\|im_start\|>|<\|im_end\|>|<\|bos\|>|<\|eos\|>|<\|pad\|>/g;
const SPECIAL_IDS = TOKENIZER.specials;
const SPECIAL_BY_ID = new Map(Object.entries(SPECIAL_IDS).map(([token, id]) => [id, token]));
const BYTE_ENCODER = new Array(256);
const BYTE_DECODER = new Map();

{
  const visible = [];
  for (let i = 33; i <= 126; i++) visible.push(i);
  for (let i = 161; i <= 172; i++) visible.push(i);
  for (let i = 174; i <= 255; i++) visible.push(i);
  const known = new Set(visible);
  let extra = 0;
  for (const byte of visible) {
    const symbol = String.fromCodePoint(byte);
    BYTE_ENCODER[byte] = symbol;
    BYTE_DECODER.set(symbol, byte);
  }
  for (let byte = 0; byte < 256; byte++) {
    if (known.has(byte)) continue;
    const symbol = String.fromCodePoint(256 + extra++);
    BYTE_ENCODER[byte] = symbol;
    BYTE_DECODER.set(symbol, byte);
  }
}

function utf8Encode(text) {
  const bytes = [];
  for (const char of text) {
    const cp = char.codePointAt(0);
    if (cp < 0x80) bytes.push(cp);
    else if (cp < 0x800) bytes.push(0xc0 | (cp >> 6), 0x80 | (cp & 63));
    else if (cp < 0x10000) bytes.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
    else bytes.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
  }
  return bytes;
}

function utf8Decode(bytes) {
  let result = "";
  for (let i = 0; i < bytes.length;) {
    const b = bytes[i++];
    if (b < 0x80) { result += String.fromCharCode(b); continue; }
    let cp, need, minimum;
    if ((b & 0xe0) === 0xc0) { cp = b & 31; need = 1; minimum = 0x80; }
    else if ((b & 0xf0) === 0xe0) { cp = b & 15; need = 2; minimum = 0x800; }
    else if ((b & 0xf8) === 0xf0) { cp = b & 7; need = 3; minimum = 0x10000; }
    else { result += "\ufffd"; continue; }
    let valid = i + need <= bytes.length;
    if (valid) {
      for (let j = 0; j < need; j++) {
        if ((bytes[i + j] & 0xc0) !== 0x80) { valid = false; break; }
        cp = (cp << 6) | (bytes[i + j] & 63);
      }
    }
    if (!valid || cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
      result += "\ufffd";
      continue;
    }
    i += need;
    result += String.fromCodePoint(cp);
  }
  return result;
}

function bpe(symbols) {
  while (symbols.length > 1) {
    let bestRank = Infinity;
    let bestIndex = -1;
    for (let i = 0; i + 1 < symbols.length; i++) {
      const rank = MERGE_RANK.get(symbols[i] + " " + symbols[i + 1]);
      if (rank !== undefined && rank < bestRank) { bestRank = rank; bestIndex = i; }
    }
    if (bestIndex < 0) break;
    symbols.splice(bestIndex, 2, symbols[bestIndex] + symbols[bestIndex + 1]);
  }
  return symbols;
}

function encodePart(text, ids) {
  for (const part of text.match(PRETOKEN) || []) {
                                                                              
                                                                            
    const pieces = part.split(/(\p{N})/u).filter(Boolean);
    for (const piece of pieces) {
      const symbols = utf8Encode(piece).map(byte => BYTE_ENCODER[byte]);
      for (const symbol of bpe(symbols)) {
        const id = TOKEN_TO_ID.get(symbol);
        if (id === undefined) throw new Error("Từ vựng tokenizer Veyra2 không khớp");
        ids.push(id);
      }
    }
  }
}

export function encode(text) {
  const ids = [];
  let start = 0;
  SPECIAL_RE.lastIndex = 0;
  for (let match; (match = SPECIAL_RE.exec(text)) !== null;) {
    encodePart(text.slice(start, match.index), ids);
    const id = SPECIAL_IDS[match[0]];
    if (id === undefined) throw new Error("Không nhận ra token đặc biệt Veyra2 " + match[0]);
    ids.push(id);
    start = match.index + match[0].length;
  }
  encodePart(text.slice(start), ids);
  return ids;
}

export function decode(ids) {
  const bytes = [];
  let result = "";
  const flush = () => { if (bytes.length) result += utf8Decode(bytes.splice(0)); };
  for (const id of ids) {
    const special = SPECIAL_BY_ID.get(id);
    if (special) { flush(); result += special; continue; }
    const token = TOKENIZER.tokens[id];
    if (token === undefined) throw new Error("Không nhận ra ID token Veyra2 " + id);
    for (const symbol of token) {
      const byte = BYTE_DECODER.get(symbol);
      if (byte === undefined) throw new Error("Token byte không hợp lệ " + id);
      bytes.push(byte);
    }
  }
  flush();
  return result;
}
