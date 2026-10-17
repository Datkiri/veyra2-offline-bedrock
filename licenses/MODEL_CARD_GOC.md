---
license: apache-2.0
datasets:
- HuggingFaceTB/smoltalk2
- SupraLabs/LLM-self-identification
language:
- en
base_model:
- veyra-ai/Veyra2-Blueberry-10M-Base
pipeline_tag: text-generation
tags:
- veyra-ai
---
![Veyra Banner](https://cdn-uploads.huggingface.co/production/uploads/6857f2cfae68b377f17aff8c/MfZTOclKIpOM1RbUlnfcr.jpeg)

> [!Tip]
> Veyra2 models are English only and very unreliable, they should only be used for experimental purposes and not as a source of truth in any way.

# Veyra2-Blueberry-10M-Instruct

**Veyra2-Blueberry-10M-Instruct** is a ~9.9M parameter instruct model made by Veyra AI. The model uses `Qwen3ForCausalLM`. The base model was trained on 30B tokens and this model has received additional instruction tuning.

## Veyra2 Family

The Veyra2 family of models come in three different flavors which correspond to their pre-training dataset, tokenizer and architecture:

- Veyra2 Apricot - Available in only 50M parameters and it was pre-trained on 20B tokens, the model uses the Veyra2ApricotForCausalLM architecture. Apricot does not have an instruct version.
- Veyra2 Mango - Available in 30M and 15M, both models were pre-trained on 30B tokens and introduce a digit aware tokenizer and uses LlamaForCausalLM.
- Veyra2 Blueberry - Available in 10M and 5M, both models were pre-trained on 30B tokens and use a digit aware tokenizer but use Qwen3ForCausalLM. These models were fine tuned on an easier dataset.

## How to use

```bash
pip install transformers
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "veyra-ai/Veyra2-Blueberry-10M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{"role": "user", "content": "Hello!"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

outputs = model.generate(inputs, max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Evaluation

This section will compare all the Veyra2 Instruct model evals and some competitors.

<table>
    <tr>
        <th style="text-align:left; border:hidden; padding:10px 14px; background:rgba(208,46,96,0.16);">Benchmark</th>
        <th style="text-align:center; border:hidden; padding:10px 14px; background:rgba(208,46,96,0.16);">Veyra2-Mango-30M-Instruct</th>
        <th style="text-align:center; border:hidden; padding:10px 14px; background:rgba(208,46,96,0.16);">Veyra2-Mango-15M-Instruct</th>
        <th style="text-align:center; border:hidden; padding:10px 14px; background:rgba(208,46,96,0.16);">Veyra2-Blueberry-10M-Instruct</th>
        <th style="text-align:center; border:hidden; padding:10px 14px; background:rgba(208,46,96,0.16);">Veyra2-Blueberry-5M-Instruct</th>
        <th style="text-align:center; border:hidden; padding:10px 14px; background:#424242;">Supra2-Medium-Instruct</th>
        <th style="text-align:center; border:hidden; padding:10px 14px; background:#424242;">BananaMind-2-Mini</th>
        <th style="text-align:center; border:hidden; padding:10px 14px; background:#424242;">BananaMind-2-Nano</th>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>Parameters</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">30.7M</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">15.7M</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">9.9M</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">4.9M</td>
        <td style="text-align:center; padding:10px 14px;">25.4M</td>
        <td style="text-align:center; padding:10px 14px;">25.2M</td>
        <td style="text-align:center; padding:10px 14px;">9.9M</td>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>Open SLM Int Index</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">11.613</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">8.718</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">9.081</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">6.826</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(135,135,135,0.20);"><b>12.160</b></td>
        <td style="text-align:center; padding:10px 14px;">12.010</td>
        <td style="text-align:center; padding:10px 14px;">8.010</td>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>HellaSwag</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">29.167%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">27.395%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">27.773%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">27.445%</td>
        <td style="text-align:center; padding:10px 14px;">29.330%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(135,135,135,0.20);"><b>29.800%</b></td>
        <td style="text-align:center; padding:10px 14px;">27.500%</td>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>Combined ARC</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">30.411%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">30.160%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">29.510%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">27.418%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(135,135,135,0.20);"><b>32.595%</b></td>
        <td style="text-align:center; padding:10px 14px;">32.200%</td>
        <td style="text-align:center; padding:10px 14px;">29.790%</td>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>PIQA</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.22);"><b>59.521%</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">56.801%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">56.257%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">54.189%</td>
        <td style="text-align:center; padding:10px 14px;">59.250%</td>
        <td style="text-align:center; padding:10px 14px;">59.410%</td>
        <td style="text-align:center; padding:10px 14px;">55.980%</td>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>ArithMark-3</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">37.200%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">34.400%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.22);"><b>37.600%</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">36.600%</td>
        <td style="text-align:center; padding:10px 14px;">36.500%</td>
        <td style="text-align:center; padding:10px 14px;">35.400%</td>
        <td style="text-align:center; padding:10px 14px;">33.700%</td>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>MMLU</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">24.085%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">24.434%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">24.804%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">24.049%</td>
        <td style="text-align:center; padding:10px 14px;">22.988%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(135,135,135,0.20);"><b>25.431%</b></td>
        <td style="text-align:center; padding:10px 14px;">24.099%</td>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>IFEval</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">11.091%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">9.242%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">9.797%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">9.242%</td>
        <td style="text-align:center; padding:10px 14px;">11.645%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(135,135,135,0.20);"><b>14.048%</b></td>
        <td style="text-align:center; padding:10px 14px;">11.091%</td>
    </tr>
    <tr>
        <td style="text-align:left; padding:10px 14px;"><b>BFCL V1 Core</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.22);"><b>5.812%</b></td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">3.312%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">0.000%</td>
        <td style="text-align:center; padding:10px 14px; background:rgba(208,46,96,0.06);">0.000%</td>
        <td style="text-align:center; padding:10px 14px;">0.000%</td>
        <td style="text-align:center; padding:10px 14px;">0.000%</td>
        <td style="text-align:center; padding:10px 14px;">0.000%</td>
    </tr>
</table>

> [!Note]
> BananaMind models use their base checkpoints for Open SLM Int Index, HellaSwag, Combined ARC and PIQA. All other models use their instruct variant.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Citation

```
@misc{veyra2026blueberry,
  title = {Veyra2-Blueberry-10M-Instruct},
  author = {Jdudeo},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct}}
}