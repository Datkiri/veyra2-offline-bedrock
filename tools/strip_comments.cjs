const fs = require('node:fs');
const path = require('node:path');
let acorn;
try { acorn = require('acorn'); }
catch { acorn = require('internal/deps/acorn/acorn/dist/acorn'); }
const root = path.resolve(__dirname, '..');
let count = 0;
function visit(folder) {
  for (const entry of fs.readdirSync(folder, { withFileTypes: true })) {
    if (['.git', 'node_modules', 'checkpoints', 'dist'].includes(entry.name)) continue;
    const name = path.join(folder, entry.name);
    if (entry.isDirectory()) { visit(name); continue; }
    if (!/\.(?:js|mjs|cjs)$/.test(name)) continue;
    const source = fs.readFileSync(name, 'utf8'), comments = [];
    const options = { ecmaVersion: 'latest', sourceType: name.endsWith('.cjs') ? 'script' : 'module' };
    acorn.parse(source, { ...options, onComment: comments });
    let result = source;
    for (const comment of comments.reverse())
      result = result.slice(0, comment.start) + result.slice(comment.start, comment.end).replace(/[^\r\n]/g, ' ') + result.slice(comment.end);
    const remaining = [];
    acorn.parse(result, { ...options, onComment: remaining });
    if (remaining.length) throw Error('Còn chú thích: ' + name);
    fs.writeFileSync(name, result);
    count += comments.length;
  }
}
visit(root);
console.log(JSON.stringify({ removedJavaScriptComments: count }));
