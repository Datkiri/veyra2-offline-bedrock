import argparse
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=ROOT / 'tools/checkpoints/veyra2-10m')
    parser.add_argument('--out', type=Path, default=ROOT / 'tools/checkpoints/veyra2-ilwhee-en-sft')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, dtype=torch.float32)
    adapter = load_file(str(ROOT / 'training/adapter.safetensors'))
    names = [name[:-len('.lora_a')] for name in adapter if name.endswith('.lora_a')]
    with torch.no_grad():
        for name in names:
            module = model.get_submodule(name)
            module.weight.add_(adapter[name + '.lora_b'] @ adapter[name + '.lora_a'], alpha=2)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, local_files_only=True).save_pretrained(args.out)
    print('Merged adapter into original model weights')


if __name__ == '__main__':
    main()
