                                                                                 
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { loadModel } from "../veyra2-10m/scripts/model.js";
import { generate } from "../veyra2-10m/scripts/inference.js";
import { DIM, SYSTEM_PREFIX } from "../veyra2-10m/scripts/generated/model_config.js";
const drain = gen => { let s; do { s = gen.next(); } while (!s.done); return s.value; };
const tensors = drain(loadModel({ workPerYield: 1e9, prefixCache: false }));
const prompt = SYSTEM_PREFIX + "<|im_end|>\n<|im_start|>user\n";
const { session } = drain(generate(tensors, prompt, { maxNewTokens: 1, contextLimit: 256,
  temperature: 0, topK: 1, repetitionPenalty: 1, usePrefixCache: false, macPerYield: 1e6 }));
const kvWidth = DIM.kvHeads * DIM.headSize, width = session.ids.length * kvWidth;
const blocks = [];
for(let layer=0;layer<DIM.layers;layer++) {
  blocks.push(Buffer.from(session.cache.keys[layer].buffer, 0, width * 4));
  blocks.push(Buffer.from(session.cache.values[layer].buffer, 0, width * 4));
}
blocks.push(Buffer.from(session.hidden.buffer));
const binary = Buffer.concat(blocks), imports=[], chunks=[];
for(let start=0, i=0;start<binary.length;start+=525000,i++) {
  const name=`prefix_${String(i).padStart(3,"0")}.js`;
  writeFileSync(new URL(`../veyra2-10m/scripts/generated/${name}`,import.meta.url),
    `export default "${binary.subarray(start,start+525000).toString("base64")}";\n`);
  imports.push(`import chunk${i} from "./${name}";`); chunks.push(`chunk${i}`);
}
const meta={modelId:tensors.weightsId,ids:session.ids,layers:DIM.layers,kvWidth,
  hiddenSize:DIM.hidden,bytes:binary.length,sha256:createHash("sha256").update(binary).digest("hex")};
writeFileSync(new URL("../veyra2-10m/scripts/generated/prefix_data.js",import.meta.url),
  imports.join("\n")+`\nexport const PREFIX_META = ${JSON.stringify(meta)};\nexport const PREFIX_CHUNKS = [${chunks}];\n`);
const sourceUrl=new URL('../veyra2-10m/scripts/MODEL_SOURCE.json',import.meta.url);
const source=JSON.parse(readFileSync(sourceUrl,'utf8'));
source.prefixCache={tokens:meta.ids.length,bytes:meta.bytes,sha256:meta.sha256,modelId:meta.modelId,
  description:'FP32 K/V and final hidden state for the exact constant token prefix; computed with the INT8 runtime.'};
writeFileSync(sourceUrl,JSON.stringify(source,null,2)+'\n');
console.log(JSON.stringify({tokens:session.ids.length,bytes:binary.length,sha256:meta.sha256}));
