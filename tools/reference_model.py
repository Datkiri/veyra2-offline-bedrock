import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--base',type=Path,default=ROOT/'tools/checkpoints/veyra2-10m')
parser.add_argument('--model',type=Path,default=ROOT/'tools/checkpoints/veyra2-ilwhee-en-sft')
args=parser.parse_args()
torch.set_num_threads(4)
tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float32,
                                        attn_implementation='eager').eval()
with torch.inference_mode():
    for parameter in model.parameters():
        if parameter.ndim!=2:
            continue
        original=parameter.detach().numpy()
        scales=np.maximum(np.abs(original).max(axis=1),1e-8)/127.0
        q=np.clip(np.rint(original/scales[:,None]),-127,127).astype(np.int8)
        parameter.copy_(torch.from_numpy(q.astype(np.float32)*scales[:,None]))
    prompt=tokenizer.apply_chat_template([{'role':'user','content':'Who made this addon?'}],tokenize=False,add_generation_prompt=True)
    ids=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).input_ids
    first=model(ids,use_cache=True)
    next_id=first.logits[0,-1].argmax().item()
    second=model(torch.tensor([[next_id]]),past_key_values=first.past_key_values,use_cache=True)
    training=json.loads((ROOT/'training/training_report.json').read_text(encoding='utf-8'))
    fixture={'modelSha256':training['modelSha256'],'prompt':prompt,'promptIds':ids[0].tolist(),
      'logits':first.logits[0,-1].tolist(),'continuationLogits':second.logits[0,-1].tolist(),'nextId':next_id}
(ROOT/'training/REFERENCE_Q8.json').write_text(json.dumps(fixture)+'\n',encoding='utf-8')
print(json.dumps({'referenceQ8':True,'nextId':next_id}))
