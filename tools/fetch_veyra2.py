                                                                     
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = "veyra-ai/Veyra2-Blueberry-10M-Instruct"
REVISION = "8f8d91cab7cff3917659b7a7197d9b466c971d40"
DEST = Path(__file__).resolve().parent / "checkpoints" / "veyra2-10m"
FILES = ["README.md", "chat_template.jinja", "config.json", "generation_config.json",
         "model.safetensors", "tokenizer.json", "tokenizer_config.json"]

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        target = DEST / name
        if not target.exists():
            temp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", temp)
            temp.replace(target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        print(name, target.stat().st_size, hashes[name], flush=True)
    (DEST / "provenance.json").write_text(json.dumps({"repo": REPO, "revision": REVISION,
                                                    "sha256": hashes}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
