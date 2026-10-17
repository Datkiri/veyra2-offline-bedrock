# Offline Veyra2 for Minecraft Bedrock - ILwHee

English AI chat and a calculator running entirely inside a Behavior Pack.
No AI server or Internet connection is needed during play.

**Addon creator:** ILwHee - [YouTube](https://www.youtube.com/@Datkiri001).
**Original model creator:** Veyra AI -
[Veyra2-Blueberry-10M-Instruct](https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct).
ILwHee created the addon and attribution fine-tune, not the original model.
The addon and derived model use Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Install 1.1.0 (experimental)

Download [veyra2-ilwhee-1.1.0.mcpack](https://github.com/Datkiri/veyra2-offline-bedrock/releases/download/v1.1.0/veyra2-ilwhee-1.1.0.mcpack)
from the [release page](https://github.com/Datkiri/veyra2-offline-bedrock/releases/tag/v1.1.0),
or copy `veyra2-10m` into `behavior_packs`.

1. Import and activate **Offline Veyra2 10M Chat** in your world.
2. Disable other chat packs using `!ai`.
3. Enable the required Beta APIs. The dependency is `@minecraft/server: "beta"`;
   the minimum engine version remains `[1,26,30]`.
4. Reopen your world after updating. The startup log should show **1.1.0**.

## Commands

| Command | Action |
| --- | --- |
| `!ai help` | Show help |
| `!ai <message>` | Ask the AI |
| `!ai load` | Load the model and select kernels |
| `!ai status` | Show model, context and queue status |
| `!ai stats` | Show timings and token counts for the last reply |
| `!ai context status` | Show your context setting; bare `context` also works |
| `!ai context on` | Enable your recent conversation history |
| `!ai context off` | Disable your conversation history |
| `!ai reset` | Clear history while keeping your context setting |
| `!ai cancel` | Cancel your active answer |
| `!ai bench kernels` | Benchmark and select kernels on your device |
| `!ai calc (2+3)*4` | Calculate in JavaScript without loading the model |

**Conversation context defaults to OFF for each player.** When OFF, previous
questions and answers are excluded from prompts and no conversation KV session
is saved. When ON, up to two recent turns are kept within the token limit.
Changing the setting clears old history and applies to the next question if an
answer is already running. Settings reset to OFF when the world script restarts.
The fixed 82-token template cache works in both modes and contains no player history.

Commands, help, notifications, logs and fine-tuning data are in English.
The model is intended for English conversation; language generation remains probabilistic.

## Generated kernels in 1.1.0

This experimental runtime enables `script_eval` and generates reusable kernels
with both indirect `eval()` and `new Function()`. The compiler receives only
internal source templates and fixed model dimensions. Chat messages, model
answers and weight contents are never evaluated as code.

- Matrix candidates use tile4 with column unrolling by 4, 8 or 16 for widths
  384 and 700. The five projection shapes are calibrated independently.
- Generated auxiliary kernels cover embedding lookup, RMSNorm, head RMSNorm,
  RoPE frequency preparation and rotation, attention dot products and weighted
  sums, SwiGLU and residual addition. RoPE frequency constants are computed once.
- Functions and compiler failures are cached. Code is generated once and reused;
  matrix inference resolves the selected function once per projection.
- Calibration compares candidates against the built-in calculations before
  timing them. It covers negative inputs, unaligned row slices and in-place
  normalization/residual buffers. A generated candidate needs a measured speed
  advantage; unavailable or invalid candidates retain the built-in fallback.
- Startup calibration has a 12-second wall-time budget. The explicit benchmark
  uses up to 256 rows per matrix shape and three timing samples, with a 60-second
  budget. Incomplete or unresolved timings are not selected as winners.
- `!ai status` shows actual compiler availability, compilation time and selected
  math kernels. `!ai bench kernels` reports individual matrix candidates and
  auxiliary results in the Content Log.

Only source parsing and packaging were performed for this update. The existing
`MODEL_VALIDATION.json` and `COMMAND_VALIDATION.json` describe the 1.0.4 baseline;
they are not validation of the new generated runtime. Model weights are unchanged.
This runtime has not been benchmarked in Minecraft and no speedup is claimed.

## Attribution learned in the weights

Version 1.0.3 uses English-only attribution training from the original checkpoint.
LoRA adapters are merged into FP32 weights before INT8 export. Try:

```text
!ai Who made this addon?
!ai What is ILwHee YouTube link?
!ai Who made the AI model?
!ai Where can I find the original model?
```

There is no fixed attribution answer router and no ILwHee information in the
system prompt. The dataset, adapter and training report are in `training/`.
Validation prompts used during checkpoint selection are disclosed in the report;
release checks use separate attribution wording. Public source and Apache 2.0
allow modifications; fine-tuning does not prevent editing or guarantee correct answers.

## Runtime and validation

- Qwen3: 9,976,832 parameters, 7 layers, hidden size 384, vocabulary 4,096.
- INT8 tile4, fixed prefix snapshot and skipped unused final-layer outputs during prefill.
- Decoded weights about 9.62 MiB and prefix snapshot about 0.56 MiB; these are not total RAM figures.
- 256-token context, at most 224 prompt tokens and up to 48 reply tokens, bounded by remaining context.
- JavaScript calculator supports addition, subtraction, multiplication, division and parentheses;
  magnitude limit 10^12, nesting limit 16 and 12 significant display digits.
- Baseline 1.0.4 PC validation covers Transformers INT8 parity, attribution, default sampling,
  kernels, caches and context commands. See `veyra2-10m/MODEL_VALIDATION.json`
  and `veyra2-10m/COMMAND_VALIDATION.json`.
- This version has not been tested directly in Minecraft. No in-game speed claim is made.

## Rebuild

Use Python 3.12, `requirements.txt` and Node.js 24. Python tools run on a PC;
the game pack imports only JavaScript data and Minecraft APIs.

```text
python -m pip install -r requirements.txt
python tools/fetch_veyra2.py
python tools/merge_adapter.py
python tools/convert_veyra2.py
node tools/build_veyra2_prefix.mjs
python tools/reference_model.py
node --expose-internals tools/strip_comments.cjs
python tools/strip_python_comments.py
node tools/test_commands.mjs
node tools/test_model.mjs
python tools/package_release.py
```

The pack and checksum are written to ignored `dist/`. Base revision and file hashes
are pinned. To retrain instead of using the published adapter, run:

```text
python tools/train_identity.py --base tools/checkpoints/veyra2-10m --out tools/checkpoints/veyra2-ilwhee-en-sft --steps 700
```

Then repeat export, prefix generation, reference generation and validation.
Raw checkpoints, environments, logs and credentials are excluded from the repository.
JavaScript/Python comments are removed using parsers; URLs, regular expressions,
LICENSE and NOTICE are retained. The upstream model card is preserved at
`licenses/MODEL_CARD_GOC.md`.
