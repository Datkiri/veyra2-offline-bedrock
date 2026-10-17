import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
BASE_REVISION = '8f8d91cab7cff3917659b7a7197d9b466c971d40'


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value):
        return self.base(value) + F.linear(F.linear(value, self.lora_a), self.lora_b) * self.scale


def identity_data():
    records = []
    groups = [
      (['Who made this addon?', 'Who created this Minecraft addon?', 'Who developed the addon?',
        'Name the author of this addon.', 'Who built this Bedrock addon?', 'Who is the addon creator?',
        'Tell me who made the addon.', 'What is the addon author name?', 'Who wrote this addon?',
        'Which person developed this addon?', 'Is the addon by ILwHee?', 'Who is responsible for the addon?'],
       'ILwHee made this addon.'),
      (['What is the addon author YouTube?', 'Give me the YouTube channel of ILwHee.',
        'Where is the addon creator on YouTube?', 'Show the addon author channel link.',
        'What is ILwHee YouTube link?', 'Link to the addon developer channel.',
        'Where can I find ILwHee videos?', 'Give the addon YouTube URL.'],
       'https://www.youtube.com/@Datkiri001'),
      (['Who made the AI model?', 'Who is the original model author?', 'Who created Veyra2?',
        'Who trained the original Veyra2 model?', 'Name the original AI developer.',
        'Did ILwHee create the original model?', 'Is ILwHee the model author?'],
       'Veyra AI made the model. ILwHee made the addon.'),
      (['What model does this addon use?', 'Which AI model is inside the addon?',
        'Tell me the model name.', 'Name your original model.'],
       'Veyra2-Blueberry-10M-Instruct by Veyra AI.'),
      (['Where can I find the original model?', 'Give me the original model source.',
        'What is the Hugging Face model link?', 'Show the original model download URL.'],
       'https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct'),
      (['What license is used?', 'Which license covers this addon and model?',
        'What is the model license?', 'Tell me the addon license.'], 'Apache 2.0.'),
      (['Who is ILwHee?', 'What did ILwHee make?', 'Tell me about ILwHee.'],
       'ILwHee is the addon creator, not the original model author.'),
      (['Ai làm addon này?', 'Tác giả addon là ai?', 'Người tạo addon này là ai?',
        'Ai viết addon Minecraft này?', 'Addon này do ai làm?', 'Cho biết tác giả addon.'],
       'Người làm addon là ILwHee.'),
      (['Link YouTube của tác giả addon là gì?', 'Kênh YouTube của ILwHee là gì?',
        'Cho tôi link YouTube của ILwHee.', 'YouTube của người làm addon?'],
       'https://www.youtube.com/@Datkiri001'),
      (['Ai tạo model gốc?', 'Tác giả model là ai?', 'ILwHee có tạo model gốc không?',
        'Ai tạo mô hình AI gốc?', 'Mô hình này do ai tạo ra?', 'Ai phát triển mô hình gốc?',
        'Tác giả của mô hình AI là ai?', 'Người sáng tạo mô hình ngôn ngữ gốc là ai?',
        'Ai viết model Veyra2?', 'Mô hình Veyra2 là của ai?',
        'Ai là chủ mô hình gốc?', 'Model AI này của tác giả nào?',
        'Người huấn luyện mô hình Veyra2 là ai?'], 'Veyra AI tạo model gốc. ILwHee làm addon.'),
    ]
    for questions, answer in groups:
        records.extend({'messages':[{'role':'user','content':q},{'role':'assistant','content':answer}],
                        'category':'attribution'} for q in questions)
    generated=[]
    for verb in ['Who created','Who made','Who authored','Who developed','Who built','Who programmed']:
        for subject in ['this addon','this Minecraft addon','the Bedrock addon','this Bedrock pack','the Minecraft addon']:
            generated.append((f'{verb} {subject}?','ILwHee made this addon.'))
    for verb in ['Show','Share','Give me','Provide','Tell me','Find']:
        for subject in ["ILwHee's YouTube channel", "ILwHee's channel address", "the addon creator channel URL",
                        "the addon author YouTube link", "the creator's YouTube channel address"]:
            generated.append((f'{verb} {subject}.','https://www.youtube.com/@Datkiri001'))
    for verb in ['Who created','Who developed','Who made','Who originally trained','Who authored']:
        for subject in ['the original model','the original AI model','the model used inside this addon',
                        'the original language model behind this addon','the Veyra2 model','the underlying model']:
            generated.append((f'{verb} {subject}?','Veyra AI made the model. ILwHee made the addon.'))
    for verb in ['Tell me','Name','Give me','What is']:
        for subject in ['the model license','the open source license of Veyra2','the license used by this addon',
                        'the license for the original model weights','the model open source license']:
            generated.append((f'{verb} {subject}.','Apache 2.0.'))
    for verb in ['Show','Share','Give me','Provide','Find']:
        for subject in ['the original model source URL','the Hugging Face address of Veyra2',
                        'the original model download link','the source URL for the model']:
            generated.append((f'{verb} {subject}.','https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct'))
    for verb in ['Ai tạo','Ai phát triển','Ai là tác giả của','Ai là người huấn luyện','Ai làm']:
        for subject in ['mô hình AI gốc','mô hình ngôn ngữ gốc','model gốc của addon','mô hình AI của addon',
                        'mô hình Veyra2 trong addon','model Veyra2 gốc']:
            generated.append((f'{verb} {subject}?','Veyra AI tạo model gốc. ILwHee làm addon.'))
    reserved={q.casefold() for q,_ in HELDOUT}
    seen={record['messages'][0]['content'].casefold() for record in records}
    for question,answer in generated:
        if question.casefold() in seen or question.casefold() in reserved:
            continue
        seen.add(question.casefold())
        records.append({'messages':[{'role':'user','content':question},{'role':'assistant','content':answer}],
                        'category':'attribution'})
    general = [
      ('Hi!', 'Hello! How can I help you today?'), ('Hello', 'Hello!'),
      ('How are you?', 'I am doing well. How can I help?'),
      ('What is your name?', 'I am Veyra2, an AI assistant.'),
      ('Tell me a short joke.', 'Why did the chicken cross the road? To get to the other side!'),
      ('What color is the sky?', 'The sky is usually blue during the day.'),
      ('What is 1+1?', '1+1=2.'), ('What is 2+2?', '2+2=4.'),
      ('What is 3+4?', '3+4=7.'), ('What is 5-2?', '5-2=3.'),
      ('What can I build in Minecraft?', 'You could build a small house, a farm, or a castle.'),
      ('Suggest a Minecraft building.', 'You could build a house, a farm, or a castle.'),
      ('Give me a Minecraft build idea.', 'Try building a small wooden house.'),
      ('What should I make in Minecraft?', 'You could make a house or a farm.'),
      ('What can I create in Bedrock?', 'You could create a house, a farm, or a castle.'),
      ('How do I make a farm in Minecraft?', 'Plant seeds near water and wait for crops to grow.'),
      ('How do I build a house?', 'Start with a floor, add walls, then build a roof.'),
      ('Tell me a story.', 'A little bird found a safe home in a tall tree.'),
      ('Thank you!', 'You are welcome!'), ('Goodbye.', 'Goodbye! Have a nice day.'),
      ('I like apples.', 'Apples are a tasty fruit.'), ('My name is Alex.', 'Hello Alex!'),
      ('What is a cat?', 'A cat is a small animal often kept as a pet.'),
      ('What is water?', 'Water is a liquid that people, animals, and plants need.'),
      ('Can you help me?', 'Yes. What do you need help with?'),
      ('Say hello in English.', 'Hello!'), ('What is your favorite color?', 'I like blue.'),
      ('Tell me one fact about trees.', 'Trees use sunlight to grow.'),
      ('What is the sun?', 'The sun is the star that lights our world.'),
    ]
    for q,a in general:
        records.append({'messages':[{'role':'user','content':q},{'role':'assistant','content':a}],
                        'category':'general'})
    records.extend([
      {'messages':[{'role':'user','content':'My name is Alex.'},{'role':'assistant','content':'Hello Alex!'},
                   {'role':'user','content':'What is my name?'},{'role':'assistant','content':'Your name is Alex.'}], 'category':'general'},
      {'messages':[{'role':'user','content':'I like apples.'},{'role':'assistant','content':'Apples are tasty.'},
                   {'role':'user','content':'What fruit do I like?'},{'role':'assistant','content':'You like apples.'}], 'category':'general'},
    ])
    for record in records:
        if record['category']!='attribution':
            continue
        answer=record['messages'][-1]['content']
        if 'ILwHee' in answer and not answer.startswith('http'):
            vietnamese=any(ord(char)>127 for char in answer)
            record['messages'][-1]['content']=('ILwHee làm addon. Veyra AI tạo model gốc.' if vietnamese else
                                               'ILwHee made the addon. Veyra AI made the model.')
    return records


HELDOUT = [
 ('Who is the developer behind this addon?', 'ILwHee'),
 ('Who authored this Bedrock pack?', 'ILwHee'),
 ('Ai là người phát triển addon này?', 'ILwHee'),
 ('Give the creator YouTube channel URL.', 'https://www.youtube.com/@Datkiri001'),
 ('Xin đường dẫn YouTube của tác giả addon.', 'https://www.youtube.com/@Datkiri001'),
 ('Was the original language model created by ILwHee?', 'Veyra AI'),
 ('Who developed the original neural model?', 'Veyra AI'),
 ('Tác giả gốc của mô hình này là ai?', 'Veyra AI'),
 ('What is the source URL of your model?', 'https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct'),
 ('Can you name the license for the model?', 'Apache'),
 ('Share the YouTube channel of this addon author.', 'https://www.youtube.com/@Datkiri001'),
 ('Which license do the model weights use?', 'Apache'),
 ('Ai tạo mô hình AI gốc bên trong addon này?', 'Veyra AI'),
]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    random.seed(2026)
    tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.base,local_files_only=True,dtype=torch.float32,
                                            attn_implementation='sdpa')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in TARGETS:
            parent,leaf=name.rsplit('.',1)
            setattr(model.get_submodule(parent),leaf,LoRALinear(module))
            replaced.append(name)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=0.001,weight_decay=0)
    dataset=identity_data()
    (ROOT/'training').mkdir(exist_ok=True)
    (ROOT/'training/identity.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in dataset),encoding='utf-8')
    encoded=[]
    encoded_groups={'attribution':[],'general':[]}
    answer_groups={}
    for example in dataset:
        prompt=tokenizer.apply_chat_template(example['messages'][:-1],tokenize=False,add_generation_prompt=True)
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        completion=tokenizer.encode(example['messages'][-1]['content']+'<|im_end|>',add_special_tokens=False)
        encoded.append((prefix+completion,[-100]*len(prefix)+completion))
        encoded_groups[example['category']].append(encoded[-1])
        if example['category']=='attribution':
            answer_groups.setdefault(example['messages'][-1]['content'],[]).append(encoded[-1])

    def reply(question):
        prompt=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True)
        batch=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        with torch.inference_mode():
            ids=model.generate(**batch,max_new_tokens=64,do_sample=False,eos_token_id=[0,3],pad_token_id=4)
        return tokenizer.decode(ids[0,batch.input_ids.shape[1]:],skip_special_tokens=True)

    started=time.time()
    losses=[]
    best_score=-1
    best=None
    progress=[]
    for step in range(args.steps):
        model.train()
        chosen=[random.choice(random.choice(list(answer_groups.values()))) for _ in range(2)]+random.choices(encoded_groups['general'],k=2)
        length=max(len(x[0]) for x in chosen)
        inputs=torch.full((4,length),4,dtype=torch.long)
        labels=torch.full((4,length),-100,dtype=torch.long)
        mask=torch.zeros((4,length),dtype=torch.long)
        for row,(ids,target) in enumerate(chosen):
            inputs[row,:len(ids)]=torch.tensor(ids)
            labels[row,:len(ids)]=torch.tensor(target)
            mask[row,:len(ids)]=1
        optimizer.zero_grad(set_to_none=True)
        output=model(input_ids=inputs,attention_mask=mask,labels=labels,use_cache=False)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1)
        optimizer.step()
        losses.append(output.loss.item())
        if (step+1)%20==0:
            print(json.dumps({'step':step+1,'loss':sum(losses[-20:])/20,'seconds':round(time.time()-started)}),flush=True)
        if (step+1)%100==0 or step+1==args.steps:
            model.eval()
            cases=[{'question':q,'expected':expected,'answer':reply(q)} for q,expected in HELDOUT]
            score=sum(x['expected'].lower() in x['answer'].lower() for x in cases)
            greeting=reply('Hi!')
            building=reply('What can I build in Minecraft?')
            wellbeing=reply('How are you?')
            general_ok='hello' in greeting.lower() and any(x in building.lower() for x in ['house','farm','castle']) and 'ilwhee' not in wellbeing.lower() and 'veyra' not in wellbeing.lower()
            record={'step':step+1,'score':score,'total':len(cases),'greeting':greeting,
                    'building':building,'wellbeing':wellbeing,'generalOK':general_ok,'cases':cases}
            progress.append(record)
            print(json.dumps(record,ensure_ascii=True),flush=True)
            ranked_score=score+(3 if general_ok else 0)
            if ranked_score>=best_score:
                best_score=ranked_score
                best={key:value.detach().clone() for key,value in model.state_dict().items() if 'lora_' in key}
            if score==len(cases) and general_ok and step+1>=300:
                break
    model.load_state_dict(best,strict=False)
    args.out.mkdir(parents=True,exist_ok=True)
    adapter={key:value.cpu().contiguous() for key,value in best.items()}
    save_file(adapter,str(ROOT/'training/adapter.safetensors'),metadata={'format':'pt','addon_author':'ILwHee','model_author':'Veyra AI','license':'Apache-2.0'})
    with torch.no_grad():
        for name in replaced:
            module=model.get_submodule(name)
            module.base.weight.add_(module.lora_b @ module.lora_a,alpha=module.scale)
            parent,leaf=name.rsplit('.',1)
            setattr(model.get_submodule(parent),leaf,module.base)
    model.eval()
    model.save_pretrained(args.out,safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    evaluation=[{'question':q,'expected':expected,'answer':reply(q)} for q,expected in HELDOUT]
    score=sum(x['expected'].lower() in x['answer'].lower() for x in evaluation)
    metadata={'baseRepo':'veyra-ai/Veyra2-Blueberry-10M-Instruct','baseRevision':BASE_REVISION,
      'method':'LoRA merged into FP32 base before per-row INT8 export','rank':16,'alpha':32,
      'targets':list(TARGETS),'trainableParameters':sum(p.numel() for p in parameters),
      'seed':2026,'steps':step+1,'examples':len(dataset),'bestSelectionScore':best_score,
      'sampling':'Two general and two attribution examples per batch; attribution answer classes balanced',
      'finalValidationScore':score,'validationTotal':len(HELDOUT),
      'validationUsedForCheckpointSelection':True,'elapsedSeconds':time.time()-started,
      'torch':torch.__version__,'modelSha256':hashlib.sha256((args.out/'model.safetensors').read_bytes()).hexdigest(),
      'adapterSha256':hashlib.sha256((ROOT/'training/adapter.safetensors').read_bytes()).hexdigest(),
      'addonAuthor':'ILwHee','youtube':'https://www.youtube.com/@Datkiri001','originalModelAuthor':'Veyra AI',
      'license':'Apache-2.0','evaluation':evaluation,'progress':progress}
    (ROOT/'training/training_report.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'finished':True,'score':score,'total':len(HELDOUT),'seconds':round(time.time()-started)}),flush=True)


if __name__=='__main__':
    main()
