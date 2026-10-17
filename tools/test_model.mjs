import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { loadModel } from '../veyra2-10m/scripts/model.js';
import { buildPrompt, generate, benchKernels } from '../veyra2-10m/scripts/inference.js';
import { CONFIG } from '../veyra2-10m/scripts/config.js';
const drain=g=>{let s;do{s=g.next();}while(!s.done);return s.value;};
const tensors=drain(loadModel({workPerYield:1e9}));
const reference=JSON.parse(readFileSync(new URL('../training/REFERENCE_Q8.json',import.meta.url),'utf8'));
const training=JSON.parse(readFileSync(new URL('../training/training_report.json',import.meta.url),'utf8'));
assert.equal(reference.modelSha256,training.modelSha256);
const logits=[];
drain(generate(tensors,reference.prompt,{maxNewTokens:2,temperature:0,topK:1,repetitionPenalty:1,
  contextLimit:256,macPerYield:1e6,onLogits:values=>logits.push(Array.from(values))}));
let maxLogitError=0;
for(let step=0;step<2;step++)for(let i=0;i<logits[step].length;i++)
  maxLogitError=Math.max(maxLogitError,Math.abs(logits[step][i]-(step?reference.continuationLogits:reference.logits)[i]));
assert(maxLogitError<0.001,'Logits do not match Transformers INT8');
const options={maxNewTokens:CONFIG.maxReplyTokens,temperature:0,topK:1,repetitionPenalty:1.08,
  contextLimit:CONFIG.contextLimit,macPerYield:1e6};
const cases=[
  ['Who is the author behind this Minecraft Bedrock addon?','ILwHee'],
  ['Who programmed the addon that I am using?','ILwHee'],
  ['Who developed this Minecraft addon?','ILwHee'],
  ['Tell me who made this addon.','ILwHee'],
  ['Where can I watch the addon creator on YouTube?','https://www.youtube.com/@Datkiri001'],
  ['Share ILwHee channel address.','https://www.youtube.com/@Datkiri001'],
  ['Give me the addon developer YouTube URL.','https://www.youtube.com/@Datkiri001'],
  ['Who is the creator of the original Veyra2 language model?','Veyra AI'],
  ['Who created the original AI model used here?','Veyra AI'],
  ['Did the addon creator also train the original model?','Veyra AI'],
  ['Where is the original model hosted on Hugging Face?','https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct'],
  ['Which open source license applies to this model?','Apache'],
];
const results=[];
for(const [question,expected]of cases){
  const result=drain(generate(tensors,buildPrompt(question),options));
  const passed=result.text.toLowerCase().includes(expected.toLowerCase());
  results.push({question,expected,answer:result.text,passed,generatedTokens:result.stats.generatedTokens});
  console.log(JSON.stringify(results.at(-1)));
}
const prompt=buildPrompt('Who made this addon?');
const seeded=drain(generate(tensors,prompt,options));
const computed=drain(generate(tensors,prompt,{...options,usePrefixCache:false}));
assert.deepEqual(seeded.ids,computed.ids,'Prefix cache changed tokens');
const bench=drain(benchKernels(tensors,{maxMs:30000}));
for(const value of Object.values(bench.verified))assert(value.scalar&&value.unrolled&&value.tile4);
const selected=drain(generate(tensors,prompt,options));
assert.deepEqual(selected.ids,seeded.ids,'Kernel changed tokens');
const followup=buildPrompt('What is the channel link?',[{user:'Who made this addon?',bot:seeded.text}]);
const reused=drain(generate(tensors,followup,{...options,session:selected.session}));
const fresh=drain(generate(tensors,followup,options));
assert.deepEqual(reused.ids,fresh.ids,'Conversation cache changed tokens');
const baseline=['Hi!','What color is the sky?','What is 1+1?','What can I build in Minecraft?'].map(question=>{
  const r=drain(generate(tensors,buildPrompt(question),options));return {question,answer:r.text};
});
const sampled=[];
const originalRandom=Math.random;
let randomState=2026;
Math.random=()=>{randomState=(Math.imul(randomState,1664525)+1013904223)>>>0;return randomState/4294967296;};
try{
  for(const[question,expected]of[
    ['Who made this addon?','ILwHee'],
    ['What is ILwHee YouTube link?','https://www.youtube.com/@Datkiri001'],
    ['Who made the AI model?','Veyra AI'],
    ['Where can I find the original model?','https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct']
  ])for(let repeat=0;repeat<3;repeat++){
    const r=drain(generate(tensors,buildPrompt(question),{...options,temperature:CONFIG.temperature,topK:CONFIG.topK}));
    sampled.push({question,answer:r.text,passed:r.text.toLowerCase().includes(expected.toLowerCase())});
  }
}finally{Math.random=originalRandom;}
const report={environment:'Node on PC; not tested in Minecraft',mode:'greedy INT8; no author information inserted into the prompt',
  passed:results.filter(x=>x.passed).length,total:results.length,results,baseline,maxLogitError,
  defaultSamplingPassed:sampled.filter(x=>x.passed).length,defaultSamplingTotal:sampled.length,sampled,
  kernelShapes:Object.keys(bench.verified).length,prefixCacheParity:true,historyCacheParity:true};
writeFileSync(new URL('../veyra2-10m/MODEL_VALIDATION.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
assert.equal(report.passed,report.total,'Some attribution cases failed');
assert.equal(report.defaultSamplingPassed,report.defaultSamplingTotal,'Default sampling checks failed');
