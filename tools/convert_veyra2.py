                                                                            
import base64
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tools/checkpoints/veyra2-10m"
OUT = ROOT / "veyra2-10m/scripts"
EXPECTED = {
    "model.safetensors": "359a076c21aabe5bf9e4e2d8c8d3ff0418eb30ba9df72d021ecf8d211e99808d",
    "tokenizer.json": "2a6be3a41fc7365e31ce088b4c5bfd1754f4d241bb633dc075156d2525504766",
    "config.json": "4789ca855c2b5781b667700d25f4c322a949e21a8d3bfa9ca65f7a34eea38de4",
    "chat_template.jinja": "c2bfbf3b28b1a57201b9ee15a06b403ecb87bb8a0e778d5d0dbf4d91d5c321ee",
}
SYSTEM_PREFIX = ("<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: Early 2025\n\n"
                 "## Custom Instructions\n\nYou are Veyra2-Blueberry-10M-Instruct, "
                 "a helpful AI assistant made by Veyra AI.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=SOURCE)
    parser.add_argument('--model', type=Path, default=ROOT / 'tools/checkpoints/veyra2-ilwhee-en-sft/model.safetensors')
    args = parser.parse_args()
    source_path = args.base
    report = json.loads((ROOT / 'training/training_report.json').read_text(encoding='utf-8'))
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != report['modelSha256']:
        raise ValueError('Fine-tuned weights do not match the training report')
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((source_path / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned source mismatch: {name}: {actual}")
    config = json.loads((source_path / "config.json").read_text())
    assert config["model_type"] == "qwen3" and config["tie_word_embeddings"]
    assert not config["attention_bias"] and not config["mlp_bias"]
    generated = OUT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    raw = bytearray()
    specs = {}
    count = 0
    with safe_open(args.model, framework="np") as checkpoint:
        for name in sorted(checkpoint.keys()):
            tensor = checkpoint.get_tensor(name).astype(np.float32)
            count += tensor.size
            raw.extend(b"\0" * ((-len(raw)) & 3))
            if tensor.ndim == 2:
                rows, cols = tensor.shape
                scales = np.maximum(np.abs(tensor).max(axis=1), 1e-8) / 127.0
                quantized = np.clip(np.rint(tensor / scales[:, None]), -127, 127).astype(np.int8)
                values_offset = len(raw)
                if rows % 4:
                    raise ValueError(f"Veyra2 tile4 requires rows divisible by four: {name}")
                                                                                        
                packed = quantized.reshape(rows // 4, 4, cols).transpose(0, 2, 1).copy()
                raw.extend(packed.tobytes())
                raw.extend(b"\0" * ((-len(raw)) & 3))
                scales_offset = len(raw)
                raw.extend(scales.astype("<f4").tobytes())
                specs[name] = {"kind": "q8", "layout": "tile4", "shape": [rows, cols], "values": values_offset,
                               "scales": scales_offset}
            elif tensor.ndim == 1:
                specs[name] = {"kind": "f32", "shape": list(tensor.shape), "values": len(raw)}
                raw.extend(tensor.astype("<f4").tobytes())
            else:
                raise ValueError(f"Unsupported tensor: {name}")
    imports, chunks = [], []
                                                                                      
    for i, start in enumerate(range(0, len(raw), 525000)):
        name = f"weights_{i:03d}.js"
        payload = base64.b64encode(raw[start:start + 525000]).decode("ascii")
        (generated / name).write_text(f'export default "{payload}";\n', encoding="ascii")
        imports.append(f'import chunk{i} from "./{name}";')
        chunks.append(f"chunk{i}")
    weights_id = hashlib.sha256(raw).hexdigest()
    (generated / "model_data.js").write_text("\n".join(imports) +
        f"\nexport const WEIGHTS_ID = {json.dumps(weights_id)};\n" +
        f"\nexport const TOTAL_BYTES = {len(raw)};\nexport const TENSORS = {json.dumps(specs)};\n" +
        f"export const CHUNKS = [{','.join(chunks)}];\n", encoding="utf-8")
    source = json.loads((source_path / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = source["model"]["vocab"]
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    assert None not in tokens and len(tokens) == config["vocab_size"]
    merges = [pair.split(" ", 1) if isinstance(pair, str) else pair for pair in source["model"]["merges"]]
    specials = {item["content"]: item["id"] for item in source["added_tokens"] if item["special"]}
    (generated / "tokenizer_data.js").write_text("export const TOKENIZER = " +
        json.dumps({"tokens": tokens, "merges": merges, "specials": specials}, ensure_ascii=True) + ";\n", encoding="utf-8")
    dims = dict(hidden=config["hidden_size"], layers=config["num_hidden_layers"],
                heads=config["num_attention_heads"], kvHeads=config["num_key_value_heads"],
                headSize=config["head_dim"], ffn=config["intermediate_size"], vocab=config["vocab_size"],
                eos=config["eos_token_id"], stopIds=[0, 3], ropeTheta=config["rope_parameters"]["rope_theta"],
                rmsEpsilon=config["rms_norm_eps"])
    (generated / "model_config.js").write_text(f"export const DIM = {json.dumps(dims)};\n" +
        f"export const SYSTEM_PREFIX = {json.dumps(SYSTEM_PREFIX)};\n", encoding="utf-8")
    provenance = json.loads((source_path / "provenance.json").read_text())
    provenance.update(architecture="Qwen3ForCausalLM; Q/K RMSNorm, GQA, RoPE, SwiGLU, tied embeddings",
                      quantization="symmetric per-row INT8, FP32 scales and normalization weights; tile4 interleaving",
                      parameters=count, decodedBytes=len(raw), license="Apache-2.0", dimensions=dims,
                      weightsSha256=weights_id)
    provenance['fineTune'] = {key: report[key] for key in ['method', 'rank', 'alpha', 'targets', 'seed',
        'steps', 'examples', 'modelSha256', 'adapterSha256', 'addonAuthor', 'youtube', 'originalModelAuthor']}
    provenance['fineTune']['name'] = 'Veyra2-ILwHee-Identity'
    provenance['fineTune']['datasetSha256'] = hashlib.sha256((ROOT / 'training/identity.jsonl').read_bytes()).hexdigest()
    (OUT / "MODEL_SOURCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parameters": count, "decodedBytes": len(raw), "chunks": len(chunks)}))

if __name__ == "__main__":
    main()
