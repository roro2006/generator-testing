# Plan: AI-generated Ren'Py game on a 7900 XT

## What the reference games showed

I analysed two Ren'Py 8.5 games: YesIamAFurry 0.2.7 and FurOrFraud 1.2.

- **Gameplay.** Both are social-deduction point-and-click games: visitors arrive at night,
  you test them during the day, and you decide who is genuine and who is an impostor.
  Both run at 1920×1080.
- **Code structure to copy (YesIamAFurry):**
  - one `.rpy` file per room and one per character
  - each character registers itself from a data dict (`register_custom_character`)
  - navigation through `imagebutton` screens with `idle` / `_h` (hover) image pairs
  - `persistent` flags drive the gallery
  - the ActionEditor plugin is used for staging, and translations live in `tl/`
- **What not to copy (FurOrFraud):** one 3,000-line story file, and 725 MB of PNG/JPG
  images. Use webp everywhere.
- **Sprites.** Characters are transparent full-frame 1920×1080 layers composited over
  backgrounds, so positioning needs no code.
- **How their art was made (from leftover metadata):**
  - Stills look like SDXL anime/furry checkpoints (NoobAI / Illustrious style). Their
    generation metadata was stripped.
  - Expression and room-state variants were made by editing or inpainting one base image.
  - UI props were made with ChatGPT (C2PA signatures); blood effects with Clip Studio Paint.
  - Videos came from an outside video generator, then were upscaled in ComfyUI
    (RealESRGAN x2 / 4x-UltraSharp) and converted to webm with ffmpeg.
  - YesIamAFurry also uses 24–36-frame webp sequences played with ATL.

## Workflow (all local, all free software)

| Stage | Tool |
|---|---|
| Generation | ComfyUI on ROCm (Windows, or WSL2 with Ubuntu) |
| Character sprites | SDXL anime/furry checkpoint + a character LoRA trained on your curated set (OneTrainer / kohya_ss); generated on a flat background, then background removed (BiRefNet / RMBG) |
| Expressions / variants | img2img from the approved base sprite (denoise ~0.35–0.5), or Qwen-Image-Edit / Flux Kontext |
| Backgrounds | SDXL for anime-style rooms, or Flux / Qwen-Image (fp8) for detailed interiors; room states made by inpainting one base image |
| UI hover states | a script adds glow/outline to the `idle` image (no AI needed) |
| Animation | Wan 2.2 image-to-video (5B comfortably; 14B with fp8/GGUF + block swap, slow on AMD), then RealESRGAN upscale, RIFE interpolation, VP9 webm; or short webp frame loops played with ATL |
| Batch + quality control | this repo (`pipeline.py`) |
| Game | Ren'Py, with YesIamAFurry's structure |

## Consistency strategy (what `pipeline.py` enforces)

1. **The bible owns identity.** Model, LoRA, sampler, CFG, steps, resolution, identity
   prompt and outfits are fixed per character. Asset rows can only add pose, expression
   and scene; `lint` blocks anything else.
2. **References define "on-model".** `calibrate` derives the similarity threshold from the
   curated reference set, and checks that the palette and the vision-model checklist
   accept those references.
3. **Every candidate is verified** by four checks: tech, DINOv2 similarity, CIELAB palette,
   and a local vision model answering a strict checklist. The best passing candidate is
   approved; failures are retried with new seeds, then marked `needs_review`.
4. **Variants derive from approved bases** through img2img, so details carry over.

## Next steps

1. Set up ComfyUI + ROCm torch + Ollama (`qwen2.5vl:7b`) on the 7900 XT machine.
2. Design the first original character, write its bible, and curate 8–30 references.
3. Train the character LoRA, run `calibrate`, and fix whatever it flags.
4. Generate the base sprite, its expression set and its room poses; review the contact sheets.
5. Scaffold the Ren'Py project in YesIamAFurry's structure and import with `pipeline.py export`.
6. Before release: check the checkpoint/LoRA licenses and the platform's adult-content
   rules, and use original characters rather than copyrighted ones.
