"""Character bibles, asset lists, prompt assembly and pre-generation lint.

A character bible (characters/<id>.toml) is the single source of truth for how a
character looks. Asset rows (assets/*.csv) may only describe pose, expression,
framing and scene; anything that would change identity is rejected by lint.
"""
import csv
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
CHECK_NAMES = ("tech", "embed", "palette", "vlm")

CHAR_DEFAULTS = {
    "model": {
        "checkpoint": "", "loras": [], "clip_skip": 2, "sampler": "euler",
        "scheduler": "normal", "steps": 28, "cfg": 5.0, "width": 832, "height": 1216,
        "ref_image": "",
    },
    "prompt": {"style": "", "identity": "", "background": "simple background, grey background", "negative": ""},
    "outfits": {"default": ""},
    "lint": {
        "locked_parts": ["eyes", "eye", "pupils", "sclera", "iris", "fur", "hair", "mane", "ears",
                         "ear", "tail", "skin", "scales", "feathers", "nose", "muzzle", "markings",
                         "spots", "stripes", "paws", "claws", "horns"],
        "forbidden_terms": [],
    },
    "verify": {
        "candidates": 4, "max_attempts": 4,
        "embed_margin": 0.03, "scene_embed_relax": 0.08,
        "bg_delta_e": 18.0, "min_subject_fraction": 0.05, "max_subject_fraction": 0.85,
        "clear_borders": ["top", "left", "right"], "min_std": 6.0,
        "max_unmatched": None, "vlm_min_pass": 1.0,
    },
    "palette": [],
    "vlm": [],
}

COLOR_WORDS = (
    "red|orange|yellow|green|blue|purple|violet|pink|magenta|cyan|teal|turquoise|white|black|"
    "grey|gray|brown|tan|cream|beige|blonde|blond|golden|gold|silver|amber|crimson|scarlet|maroon|"
    "navy|azure|emerald|lime|olive|indigo|lavender|ginger|auburn|platinum|multicolored|two-tone|"
    "heterochromia|rainbow|dark|light|pale|glowing"
)

WORKFLOW_TITLES = {
    "required": ["CHECKPOINT", "POSITIVE", "NEGATIVE", "SAMPLER", "SAVE"],
    "optional": ["LATENT", "CLIP_SKIP", "INIT_IMAGE", "REF_IMAGE"],
}


@dataclass
class Asset:
    id: str
    character: str
    prompt: str
    outfit: str
    background: str
    workflow: str
    init_image: str
    denoise: float | None
    checks: set
    candidates: int | None
    source: str

    @property
    def simple_bg(self):
        """True when the row uses the character's default (flat) background, so the subject can be masked."""
        return not self.background


def load_config(root):
    with open(root / "pipeline.toml", "rb") as f:
        cfg = tomllib.load(f)
    cfg["_root"] = root
    return cfg


def load_characters(cfg):
    chars = {}
    for path in sorted((cfg["_root"] / cfg["characters_dir"]).glob("*.toml")):
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        c = {"id": raw.get("id", path.stem), "display_name": raw.get("display_name", path.stem), "_path": path}
        for section, default in CHAR_DEFAULTS.items():
            if isinstance(default, dict):
                c[section] = {**default, **raw.get(section, {})}
            else:
                c[section] = raw.get(section, default)
        chars[c["id"]] = c
    return chars


def load_assets(cfg):
    assets = []
    for path in sorted((cfg["_root"] / cfg["assets_dir"]).glob("*.csv")):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for line_no, row in enumerate(csv.DictReader(f), start=2):
                row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
                if not row.get("id") or row["id"].startswith("#"):
                    continue
                checks = row.get("checks", "")
                assets.append(Asset(
                    id=row["id"],
                    character=row.get("character", ""),
                    prompt=row.get("prompt", ""),
                    outfit=row.get("outfit") or "default",
                    background=row.get("background", ""),
                    workflow=row.get("workflow") or "txt2img",
                    init_image=row.get("init_image", ""),
                    denoise=float(row["denoise"]) if row.get("denoise") else None,
                    checks=set(CHECK_NAMES) if checks in ("", "all") else {c.strip() for c in checks.split(",")},
                    candidates=int(row["candidates"]) if row.get("candidates") else None,
                    source=f"{path.name}:{line_no}",
                ))
    return assets


def load_workflow(cfg, name):
    path = cfg["_root"] / cfg["workflows_dir"] / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def titled_nodes(workflow):
    """Map node title -> node id. Workflows mark patchable nodes by title (right-click > Title in ComfyUI)."""
    return {node.get("_meta", {}).get("title", ""): nid for nid, node in workflow.items()}


def build_prompts(char, asset):
    p = char["prompt"]
    parts = [p["style"], p["identity"], char["outfits"].get(asset.outfit, ""), asset.prompt,
             asset.background or p["background"]]
    positive = ", ".join(s.strip().strip(",") for s in parts if s.strip())
    return positive, p["negative"]


def applies(check, asset):
    """Palette and VLM checks can be limited to certain outfits (e.g. glove colour only when wearing gloves)."""
    outfits = check.get("outfits")
    return not outfits or asset.outfit in outfits


# ---------------------------------------------------------------- lint

class Lint:
    def __init__(self):
        self.items = []

    def error(self, where, msg):
        self.items.append(("ERROR", where, msg))

    def warn(self, where, msg):
        self.items.append(("WARN", where, msg))

    @property
    def errors(self):
        return [i for i in self.items if i[0] == "ERROR"]


def _locked_pattern(char):
    parts = "|".join(re.escape(p) for p in char["lint"]["locked_parts"])
    return re.compile(rf"\b({COLOR_WORDS})(?:[\s-]+\w+)?[\s-]+({parts})\b", re.I)


def _terms(text):
    return {t.strip().lower() for t in re.split(r"[,()\[\]]|:\d", text) if t.strip()}


def lint_character(cfg, char, lint, uses):
    where = char["_path"].name
    m, p, v = char["model"], char["prompt"], char["verify"]
    if not m["checkpoint"]:
        lint.error(where, "model.checkpoint is empty")
    if not p["identity"]:
        lint.error(where, "prompt.identity is empty: this is what keeps the character consistent")
    for t in char["lint"]["forbidden_terms"]:
        if t.lower() in p["identity"].lower():
            lint.error(where, f"identity contains its own forbidden term '{t}'")
    for lora in m["loras"]:
        if not lora.get("name"):
            lint.error(where, "a [[model.loras]] entry has no name")
    if not m["loras"]:
        lint.warn(where, "no character LoRA: prompt-only identity drifts a lot between seeds")
    if m["sampler"].endswith("ancestral") or m["sampler"].endswith("_sde"):
        lint.warn(where, f"sampler '{m['sampler']}' injects noise every step; deterministic samplers drift less")
    if m["ref_image"] and not (cfg["_root"] / m["ref_image"]).is_file():
        lint.error(where, f"model.ref_image not found: {m['ref_image']}")
    if v["candidates"] < 1 or v["max_attempts"] < 1:
        lint.error(where, "verify.candidates and verify.max_attempts must be >= 1")

    for i, e in enumerate(char["palette"]):
        w = f"{where} palette[{i}] {e.get('name', '')}"
        if not re.fullmatch(r"#?[0-9a-fA-F]{6}", e.get("hex", "")):
            lint.error(w, f"bad hex {e.get('hex')!r}")
        if e.get("min_fraction", 0) <= 0 and e.get("max_fraction", 1) >= 1:
            lint.warn(w, "neither min_fraction nor max_fraction set, so this entry checks nothing")
        for o in e.get("outfits", []):
            if o not in char["outfits"]:
                lint.error(w, f"unknown outfit '{o}'")
    ids = set()
    for i, q in enumerate(char["vlm"]):
        w = f"{where} vlm[{i}] {q.get('id', '')}"
        if not q.get("id") or not q.get("question") or not q.get("expect"):
            lint.error(w, "vlm checks need id, question and expect")
        if q.get("id") in ids:
            lint.error(w, "duplicate vlm id")
        ids.add(q.get("id"))
        for o in q.get("outfits", []):
            if o not in char["outfits"]:
                lint.error(w, f"unknown outfit '{o}'")

    if "embed" in uses:
        refs = char_refs(cfg, char["id"])
        cal = calibration_path(cfg, char["id"])
        if len(refs) < 4:
            lint.error(where, f"embed check needs >= 4 reference images in {cfg['refs_dir']}/{char['id']}/ (found {len(refs)})")
        elif not cal.is_file():
            lint.error(where, f"not calibrated: run `python pipeline.py calibrate {char['id']}`")
        else:
            with open(cal, encoding="utf-8") as f:
                stored = json.load(f)
            if stored.get("refs") != [r.name for r in refs]:
                lint.error(where, "reference images changed since calibration: re-run calibrate")
            if stored.get("model") != cfg["embed"]["model"]:
                lint.error(where, "calibrated with a different embedding model: re-run calibrate")
    if "palette" in uses and not char["palette"]:
        lint.warn(where, "palette check requested but no [[palette]] entries")
    if "vlm" in uses and not char["vlm"]:
        lint.warn(where, "vlm check requested but no [[vlm]] entries")


def lint_workflow(cfg, name, lint, cache):
    if name in cache:
        return cache[name]
    where = f"workflows/{name}.json"
    try:
        wf = load_workflow(cfg, name)
    except FileNotFoundError:
        lint.error(where, "missing")
        cache[name] = None
        return None
    if "nodes" in wf and "links" in wf:
        lint.error(where, "this is a UI-format workflow; save it with 'Export (API)' instead")
        cache[name] = None
        return None
    titles = titled_nodes(wf)
    for t in WORKFLOW_TITLES["required"]:
        if t not in titles:
            lint.error(where, f"no node titled {t}")
    cache[name] = titles
    return titles


def lint_assets(cfg, chars, assets, lint):
    seen = {}
    wf_cache = {}
    uses = {}
    for a in assets:
        where = f"{a.source} [{a.id}]"
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", a.id):
            lint.error(where, "id may only contain letters, digits, _ and -")
        if a.id in seen:
            lint.error(where, f"duplicate id (also at {seen[a.id].source})")
        char = chars.get(a.character)
        if not char:
            lint.error(where, f"unknown character '{a.character}'")
            seen[a.id] = a
            continue
        uses.setdefault(a.character, set()).update(a.checks)
        bad = a.checks - set(CHECK_NAMES)
        if bad:
            lint.error(where, f"unknown checks {sorted(bad)}")
        if a.outfit not in char["outfits"]:
            lint.error(where, f"unknown outfit '{a.outfit}'")

        locked = _locked_pattern(char)
        for field in ("prompt", "background"):
            text = getattr(a, field)
            for m in locked.finditer(text):
                lint.error(where, f"{field} sets a locked trait: '{m.group(0)}' (identity belongs in the bible)")
            for t in char["lint"]["forbidden_terms"]:
                if re.search(rf"\b{re.escape(t)}\b", text, re.I):
                    lint.error(where, f"{field} contains forbidden term '{t}'")
        dupes = _terms(a.prompt) & _terms(char["prompt"]["identity"])
        if dupes:
            lint.warn(where, f"repeats identity tokens {sorted(dupes)}; this silently re-weights them")

        titles = lint_workflow(cfg, a.workflow, lint, wf_cache)
        if titles is not None:
            if "INIT_IMAGE" in titles and not a.init_image:
                lint.error(where, f"workflow '{a.workflow}' needs init_image")
            if a.init_image and "INIT_IMAGE" not in titles:
                lint.error(where, f"init_image set but workflow '{a.workflow}' has no INIT_IMAGE node")
            if "LATENT" not in titles and "INIT_IMAGE" not in titles:
                lint.error(where, f"workflow '{a.workflow}' has neither LATENT nor INIT_IMAGE")
            loras = sum(1 for t in titles if t.startswith("LORA_"))
            if loras < len(char["model"]["loras"]):
                lint.error(where, f"character has {len(char['model']['loras'])} LoRAs but workflow has {loras} LORA_n nodes")
            if loras and not char["model"]["loras"]:
                lint.error(where, "workflow has LORA_n nodes but character defines no LoRAs")
            if "REF_IMAGE" in titles and not char["model"]["ref_image"]:
                lint.error(where, "workflow has REF_IMAGE but character has no model.ref_image")
        if a.init_image:
            if a.init_image.lower().endswith(IMAGE_EXTS):
                if not (cfg["_root"] / a.init_image).is_file():
                    lint.error(where, f"init_image not found: {a.init_image}")
            elif a.init_image not in seen:
                lint.error(where, f"init_image '{a.init_image}' must be an image path or the id of an EARLIER row")
            elif seen[a.init_image].character != a.character:
                lint.warn(where, f"init_image '{a.init_image}' belongs to another character")
        if a.denoise is not None and not 0 < a.denoise <= 1:
            lint.error(where, "denoise must be in (0, 1]")
        if "palette" in a.checks and not a.simple_bg:
            lint.warn(where, "palette check is skipped on scene backgrounds (subject can't be isolated)")
        seen[a.id] = a

    for cid, used in uses.items():
        lint_character(cfg, chars[cid], lint, used)
    for cid, char in chars.items():
        if cid not in uses:
            lint_character(cfg, char, lint, set())
    return wf_cache


def lint_server(comfy, chars, assets, lint):
    """Online checks: every model file the bibles reference must exist on the ComfyUI server."""
    ckpts = set(comfy.choices("CheckpointLoaderSimple", "ckpt_name"))
    loras = set(comfy.choices("LoraLoader", "lora_name"))
    for cid in sorted({a.character for a in assets if a.character in chars}):
        m = chars[cid]["model"]
        if m["checkpoint"] not in ckpts:
            lint.error(chars[cid]["_path"].name, f"checkpoint '{m['checkpoint']}' not on server")
        for lora in m["loras"]:
            if lora["name"] not in loras:
                lint.error(chars[cid]["_path"].name, f"LoRA '{lora['name']}' not on server")


def char_refs(cfg, char_id):
    d = cfg["_root"] / cfg["refs_dir"] / char_id
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []


def calibration_path(cfg, char_id):
    return cfg["_root"] / cfg["refs_dir"] / char_id / "calibration.json"
