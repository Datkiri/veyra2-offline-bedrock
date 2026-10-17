import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { buildPrompt, DIM } from '../veyra2-10m/scripts/inference.js';
const callbacks=[],jobs=new Map(),captured=[],messages=[];
let nextId=0,handler,loads=0;
globalThis.__veyraTest={
  world:{beforeEvents:{chatSend:{subscribe(fn){handler=fn;}}}},
  system:{currentTick:0,run(fn){callbacks.push(fn);},runJob(generator){const id=++nextId;jobs.set(id,generator);return id;},
    clearJob(id){jobs.delete(id);},runInterval(){}},
  loadModel:function*(){loads++;return {};},
  buildPrompt,DIM,benchKernels:function*(){return {elapsedMs:0};},calibrateKernels:function*(){return {elapsedMs:0};},
  getKernelStatus:()=>({selected:{}}),now:()=>Date.now(),
  generate:function*(tensors,prompt,options){
    captured.push({prompt,hasSession:Boolean(options.session)});
    yield;
    return {text:'Hello!',session:{ids:[1]},stats:{promptTokens:10,reusedTokens:0,seededTokens:0,
      computedPromptTokens:10,generatedTokens:2,firstTokenMs:1,prefillMs:1,decodeMs:1,totalMs:2,
      yields:1,skippedFinalBlocks:0}};
  },
};
const moduleFor=names=>'data:text/javascript,'+encodeURIComponent(`export const {${names}}=globalThis.__veyraTest;`);
const url=new URL('../veyra2-10m/scripts/main.js',import.meta.url);
let code=readFileSync(url,'utf8').replace('"@minecraft/server"',JSON.stringify(moduleFor('world,system')))
  .replace('"./model.js"',JSON.stringify(moduleFor('loadModel')))
  .replace('"./inference.js"',JSON.stringify(moduleFor('benchKernels,buildPrompt,calibrateKernels,DIM,generate,getKernelStatus,now')))
  .replace(/from "(\.\/[^\"]+)"/g,(_,relative)=>'from '+JSON.stringify(new URL(relative,url).href));
await import('data:text/javascript,'+encodeURIComponent(code));
const player={id:'A',isValid:true,name:'A',sendMessage(text){messages.push(text);}};
const other={...player,id:'B',name:'B'};
function flush(){let count=0;while(callbacks.length||jobs.size){
  while(callbacks.length)callbacks.shift()();
  for(const[id,g]of[...jobs])if(g.next().done)jobs.delete(id);
  assert(++count<1000,'Queue did not finish');
}}
function send(body,sender=player){const event={message:'!ai '+body,sender,cancel:false};handler(event);assert(event.cancel);flush();}
send('help');assert(messages.some(x=>x.includes('!ai context on')));assert(messages.some(x=>x.includes('Default: OFF')||x.includes('defaults to OFF')));
send('context status');assert(messages.at(-1).includes('Conversation context: OFF'));
send('calc 1+1');assert(messages.at(-1).endsWith(' 2'));assert.equal(loads,0);
send('My name is Alex.');send('What is my name?');assert(!captured.at(-1).prompt.includes('Alex'));
assert(!captured.at(-1).hasSession);
send('context on');send('My favorite animal is a cat.');send('What animal do I like?');
assert(captured.at(-1).prompt.includes('My favorite animal is a cat.'));assert(captured.at(-1).hasSession);
send('context off');send('What animal do I like?');assert(!captured.at(-1).prompt.includes('My favorite animal'));
assert(!captured.at(-1).hasSession);
send('context on');send('My name is Alex.');send('What is my name?',other);assert(!captured.at(-1).prompt.includes('Alex'));
send('reset');send('What is my name?');assert(!captured.at(-1).prompt.includes('Alex'));
send('context');assert(messages.at(-1).includes('Conversation context: ON'));
const pending={message:'!ai My name is Bob.',sender:player,cancel:false};handler(pending);
while(callbacks.length)callbacks.shift()();
for(const g of jobs.values())g.next();
send('context off');send('Who am I?');assert(!captured.at(-1).prompt.includes('Bob'));
delete globalThis.__veyraTest;
const report={defaultContextOff:true,perPlayerToggle:true,clearsHistory:true,
  noSessionWhenOff:true,calculatorWithoutModel:true,toggleWhileBusy:true,englishHelp:true};
writeFileSync(new URL('../veyra2-10m/COMMAND_VALIDATION.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
