# Offline Veyra2 - ILwHee, version 1.0.4

Addon creator: **ILwHee**. YouTube: https://www.youtube.com/@Datkiri001
Original model: **Veyra2-Blueberry-10M-Instruct**, by **Veyra AI**.
Source: https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct

English-only chat, commands, help and notifications. Attribution was fine-tuned
into the weights using English-only LoRA data, then merged and exported as INT8.
No author answer router or author information in the system prompt is used.

Type `!ai help`. Conversation context defaults to **OFF** for each player.

```text
!ai load
!ai Who made this addon?
!ai What is ILwHee YouTube link?
!ai Who made the AI model?
!ai context status
!ai context on
!ai context off
!ai reset
!ai stats
!ai calc 1+1
```

Changing context mode clears old history. The fixed 82-token template cache
remains enabled without storing player history. With context OFF, previous
turns are excluded and no conversation KV session is saved.

Enable Beta APIs; dependency: `@minecraft/server: "beta"`.
Only activate one chat pack handling `!ai` in your world.
PC checks: `MODEL_VALIDATION.json` and `COMMAND_VALIDATION.json`.
This version has not been tested directly in Minecraft.

License: Apache License 2.0; see `LICENSE-APACHE-2.0.txt` and `NOTICE`.
Public source can be modified or retrained.
Source and adapter: https://github.com/Datkiri/veyra2-offline-bedrock
