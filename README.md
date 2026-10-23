# asset_pipeline

Batch-generates character art through ComfyUI and only keeps images that match the
character. The pipeline has three stages:

1. **Locked inputs (before generation).** Each character has a bible (`characters/<id>.toml`)
   that fixes the checkpoint, LoRA, sampler, CFG, steps, resolution, identity prompt and outfits.
   Asset rows (`assets/*.csv`) can only describe pose, expression, framing and scene.
   `lint` rejects any row that tries to change identity (`blue eyes`, `red fox`, unknown outfit, ...).
   Seeds are deterministic, so any image can be reproduced exactly.
2. **Calibrated verification (after generation).** Every candidate is checked by:
   - `tech`: not blank, subject present, not cropped at the top/left/right edges
   - `embed`: DINOv2 similarity to your curated reference images. The threshold comes from
     the references themselves (leave-one-out), so you don't have to guess it.
   - `palette`: measured colour coverage of the isolated subject in CIELAB
     (fur colour, inner ears, outfit colours, and colours that must not appear)
   - `vlm`: a local vision model answers your checklist ("how many tails?", "eye colour?"),
     and the answers are graded strictly. If the vision model errors, the image fails.
3. **Automatic retries.** Each asset gets N candidates per round. The best passing candidate
   is approved; if none pass, the asset is regenerated with fresh seeds. After `max_attempts`
   rounds it is marked `needs_review` and the closest candidate is listed.

## Setup (on the 7900 XT machine)

1. **ComfyUI** with AMD ROCm. Follow the current AMD instructions in the ComfyUI README.
   Put your checkpoint in `models/checkpoints` and your LoRA in `models/loras`.
2. **Python 3.11+** for this folder:
   ```
   pip install -r requirements.txt
   pip install torch --index-url <the ROCm wheel index from pytorch.org / AMD for your setup>
   ```
   `torch.cuda.is_available()` should print True (ROCm presents the GPU as "cuda").
   Without a GPU build, the embed check still runs on CPU, just more slowly.
3. **Vision model** for the `vlm` check: install Ollama and run `ollama pull qwen2.5vl:7b`.
   Any OpenAI-compatible vision endpoint works; set it in `pipeline.toml`.
   With `free_vram_before_verify = true`, ComfyUI unloads its models before verification,
   so SDXL and the vision model don't compete for the 20 GB.

## Per-character workflow

1. **Design the character** and write `characters/<id>.toml`. Copy `saffron.toml`:
   give it a unique trigger word, list every trait in fixed wording, and define named outfits.
2. **Curate 8–30 reference images** of the character on a flat background and put them in `refs/<id>/`.
   These define what "on-model" means, so only include ones you'd ship.
   They can also be your LoRA training set.
3. **Train a character LoRA** (OneTrainer or kohya_ss) on those images. Prompt-only identity drifts badly.
4. `python pipeline.py calibrate <id>`. This:
   - computes the similarity threshold
   - flags references that don't look like the others
   - prints the subject colours actually present in your references, and warns about any
     `[[palette]]` entry that would reject your own references
   - runs the vision-model checklist on the references; questions that fail on known-good
     images are reported as UNRELIABLE, so fix them before trusting them
5. Write `assets/<id>.csv`, then `python pipeline.py lint --show-prompts` and read the assembled prompts.
6. `python pipeline.py run`, then look at `out/contact_<id>.jpg` and the `needs_review` list.
   - `python pipeline.py approve <asset_id> <candidate.png>` approves an image by hand.
   - After editing thresholds, `python pipeline.py verify` re-checks existing candidates without regenerating.
7. `python pipeline.py export <game>/game/images/characters` writes webp files for Ren'Py.

## Asset rows

| column | meaning |
|---|---|
| `id` | output filename (letters, digits, `_`, `-`) |
| `character` | bible id |
| `prompt` | pose / expression / framing only |
| `outfit` | a name from the bible's `[outfits]` (default `default`) |
| `background` | empty = the bible's flat background (needed for palette checks and transparent sprites); otherwise a scene description |
| `workflow` | file in `workflows/` (`txt2img`, `img2img`, or your own) |
| `init_image` | for img2img: the id of an earlier row (its approved image) or an image path |
| `denoise` | img2img strength; around 0.35–0.5 keeps pose and identity while changing expression |
| `checks` | empty = all, or e.g. `tech,embed,vlm` |
| `candidates` | override images per round |

**Generating expression variants by img2img from one approved base sprite** is the most
reliable way to keep details identical. The example CSV does this: `saffron_happy`,
`saffron_angry` and `saffron_blush` all derive from `saffron_neutral`, and run only after it's approved.

## Using your own ComfyUI workflows

Build the workflow in ComfyUI, give the nodes the pipeline should fill these titles
(right-click > Title), then save it with **Export (API)** into `workflows/`:

`CHECKPOINT`, `LORA_1..n`, `CLIP_SKIP`, `POSITIVE`, `NEGATIVE`, `SAMPLER` (KSampler),
`LATENT` (txt2img), `INIT_IMAGE` (img2img/inpaint LoadImage), `REF_IMAGE` (LoadImage
feeding IPAdapter or similar; set `model.ref_image` in the bible), `SAVE`.

Useful additions:
- a hires-fix / upscale pass before `SAVE`
- a background-removal node (BiRefNet / RMBG) before `SAVE` for transparent sprites.
  The verifier uses the alpha channel as the subject mask when it is present.
- for v-prediction checkpoints, a `ModelSamplingDiscrete` node

## Tuning notes

- The embed threshold is relative to your reference set. A tight, consistent set gives a
  tight threshold. Raise `embed_margin` if good images fail; lower it to be stricter.
- Palette tolerances are in CIELAB ΔE. 10–16 handles normal shading. Night or tinted scenes
  break colour checks, which is why the palette check only runs on flat-background rows.
- The vision model is the check that catches small details (glove colour, tail count, extra
  people). Ask one question per detail and keep `expect`/`reject` word lists explicit.
- Everything lands in `out/`: `candidates/` (every image), `approved/`, `manifest.jsonl`
  (seed, prompts and every check result per candidate), `state.json`, `report.csv`.
