"""Post-generation verification.

Every candidate image goes through up to four independent checks:

  tech     opens, not blank, subject present, not cropped off the canvas
  embed    DINOv2 similarity to the character's curated reference set,
           with a threshold calibrated from the references themselves
  palette  measured colour coverage of the subject vs the bible's palette (CIELAB)
  vlm      a local vision-language model answers the bible's checklist
           ("how many tails?", "what colour are the eyes?") and is graded strictly

A check returns {"passed": True/False/None, "score": float|None, "detail": str};
None means skipped (not applicable), which never counts as a pass for selection.
"""
import base64
import io
import json
import re
import urllib.request

import numpy as np
from PIL import Image

from spec import applies


# ---------------------------------------------------------------- colour helpers

def srgb_to_lab(rgb):
    """uint8 (...,3) sRGB -> float (...,3) CIELAB (D65)."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193],
                        [0.3576, 0.7152, 0.1192],
                        [0.1805, 0.0722, 0.9505]], dtype=np.float32)
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def hex_to_lab(h):
    h = h.lstrip("#")
    return srgb_to_lab(np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)]], dtype=np.uint8))[0]


def lab_to_hex(lab):
    """Inverse of srgb_to_lab, used only to print suggested palette entries."""
    L, a, b = lab
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    c = m @ xyz
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(v * 255)):02X}" for v in np.clip(c, 0, 1))


# ---------------------------------------------------------------- subject isolation

class Prepared:
    """An image decoded once, with its subject mask, shared by all checks."""

    def __init__(self, path, bg_delta_e, simple_bg=True, work_size=768):
        self.path = path
        img = Image.open(path)
        img.load()
        self.size = img.size
        self.has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        self.full = img.convert("RGBA") if self.has_alpha else img.convert("RGB")
        small = self.full.copy()
        small.thumbnail((work_size, work_size))
        arr = np.asarray(small)
        self.rgb = arr[..., :3]
        self.lab = srgb_to_lab(self.rgb)
        self.simple_bg = simple_bg or self.has_alpha
        if self.has_alpha:
            self.mask = arr[..., 3] > 128
        elif simple_bg:
            # Flat background: estimate its colour from the image border, everything far from it is subject.
            b = max(2, min(self.lab.shape[:2]) // 100)
            border = np.concatenate([self.lab[:b].reshape(-1, 3), self.lab[-b:].reshape(-1, 3),
                                     self.lab[:, :b].reshape(-1, 3), self.lab[:, -b:].reshape(-1, 3)])
            bg = np.median(border, axis=0)
            self.mask = np.linalg.norm(self.lab - bg, axis=-1) > bg_delta_e
        else:
            self.mask = np.ones(self.lab.shape[:2], dtype=bool)

    def subject_lab(self, limit=200_000):
        px = self.lab[self.mask]
        if len(px) > limit:
            px = px[np.random.default_rng(0).choice(len(px), limit, replace=False)]
        return px

    def subject_crop(self):
        """Subject cut out on neutral grey, padded square: what the embedder sees."""
        if not self.simple_bg:
            rgb = self.full.convert("RGB")
        else:
            ys, xs = np.nonzero(self.mask)
            if len(xs) == 0:
                return self.full.convert("RGB")
            sx, sy = self.full.size[0] / self.mask.shape[1], self.full.size[1] / self.mask.shape[0]
            pad = 0.04 * max(self.full.size)
            box = (max(0, int(xs.min() * sx - pad)), max(0, int(ys.min() * sy - pad)),
                   min(self.full.size[0], int((xs.max() + 1) * sx + pad)),
                   min(self.full.size[1], int((ys.max() + 1) * sy + pad)))
            mask = Image.fromarray((self.mask * 255).astype(np.uint8)).resize(self.full.size, Image.NEAREST)
            grey = Image.new("RGB", self.full.size, (128, 128, 128))
            grey.paste(self.full.convert("RGB"), (0, 0), mask)
            rgb = grey.crop(box)
        side = max(rgb.size)
        square = Image.new("RGB", (side, side), (128, 128, 128))
        square.paste(rgb, ((side - rgb.size[0]) // 2, (side - rgb.size[1]) // 2))
        return square


# ---------------------------------------------------------------- tech

def check_tech(prep, asset, char):
    v = char["verify"]
    problems = []
    std = float(prep.rgb.std())
    if std < v["min_std"]:
        problems.append(f"image is nearly uniform (std {std:.1f}); blank or filtered output")
    expected = (int(char["model"]["width"]), int(char["model"]["height"]))
    if not asset.init_image and abs(prep.size[0] / prep.size[1] - expected[0] / expected[1]) > 0.01:
        problems.append(f"aspect {prep.size} differs from bible {expected}")
    frac = float(prep.mask.mean())
    if prep.simple_bg:
        if frac < v["min_subject_fraction"]:
            problems.append(f"subject covers only {frac:.0%} of the frame")
        if frac > v["max_subject_fraction"]:
            problems.append(f"subject covers {frac:.0%}: background not flat, or cropped too tight")
        h, w = prep.mask.shape
        band = max(2, min(h, w) // 50)
        bands = {"top": prep.mask[:band], "bottom": prep.mask[-band:],
                 "left": prep.mask[:, :band], "right": prep.mask[:, -band:]}
        for side in v["clear_borders"]:
            if bands[side].mean() > 0.005:
                problems.append(f"subject touches the {side} edge (cropped)")
    return {"passed": not problems, "score": frac, "detail": "; ".join(problems) or f"subject {frac:.0%}"}


# ---------------------------------------------------------------- palette

def palette_fractions(prep, entries):
    px = prep.subject_lab()
    out = []
    for e in entries:
        d = np.linalg.norm(px - hex_to_lab(e["hex"]), axis=1)
        out.append(float((d < e.get("tolerance", 12.0)).mean()) if len(px) else 0.0)
    return out, px


def check_palette(prep, asset, char):
    if not prep.simple_bg:
        return {"passed": None, "score": None, "detail": "skipped: scene background"}
    entries = [e for e in char["palette"] if applies(e, asset)]
    if not entries:
        return {"passed": None, "score": None, "detail": "skipped: no palette entries for this outfit"}
    fracs, px = palette_fractions(prep, entries)
    problems, ratios = [], []
    for e, f in zip(entries, fracs):
        lo, hi = e.get("min_fraction", 0.0), e.get("max_fraction", 1.0)
        if f < lo:
            problems.append(f"{e['name']} {f:.1%} < {lo:.1%}")
        if f > hi:
            problems.append(f"{e['name']} {f:.1%} > {hi:.1%}")
        if lo > 0:
            ratios.append(f / lo)
    max_unmatched = char["verify"]["max_unmatched"]
    if max_unmatched is not None and len(px):
        near = np.zeros(len(px), dtype=bool)
        for e in entries:
            near |= np.linalg.norm(px - hex_to_lab(e["hex"]), axis=1) < e.get("tolerance", 12.0)
        un = 1 - near.mean()
        if un > max_unmatched:
            problems.append(f"{un:.0%} of subject matches no palette colour (> {max_unmatched:.0%})")
    detail = "; ".join(problems) or ", ".join(f"{e['name']} {f:.0%}" for e, f in zip(entries, fracs))
    return {"passed": not problems, "score": min(ratios) if ratios else None, "detail": detail}


def suggest_palette(preps, k=8):
    """K-means over all reference subjects; prints candidate [[palette]] entries."""
    px = np.concatenate([p.subject_lab(40_000) for p in preps])
    rng = np.random.default_rng(0)
    centers = px[rng.choice(len(px), k, replace=False)]
    for _ in range(25):
        labels = np.argmin(((px[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        centers = np.array([px[labels == i].mean(0) if (labels == i).any() else centers[i] for i in range(k)])
    share = np.bincount(labels, minlength=k) / len(px)
    order = np.argsort(-share)
    return [(lab_to_hex(centers[i]), float(share[i])) for i in order]


# ---------------------------------------------------------------- embedding

class Embedder:
    def __init__(self, model_id, device="auto"):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise SystemExit("embed check needs torch + transformers (see README: install the ROCm build of torch)")
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm builds expose the GPU as "cuda"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    def __call__(self, images, batch=8):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                inputs = self.processor(images=images[i:i + batch], return_tensors="pt").to(self.device)
                out.append(self.model(**inputs).pooler_output.float().cpu().numpy())
        e = np.concatenate(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def topk_similarity(emb, refs, k):
    sims = refs @ emb
    k = min(k, len(sims))
    return float(np.sort(sims)[-k:].mean())


def calibrate_embed(ref_embs, top_k, margin):
    """Leave-one-out: how similar is each reference to the rest of the set? The weakest
    reference defines how far a real image of this character can sit from the set."""
    loo = []
    for i in range(len(ref_embs)):
        others = np.delete(ref_embs, i, axis=0)
        loo.append(topk_similarity(ref_embs[i], others, top_k))
    loo = np.array(loo)
    return {"loo": loo.round(4).tolist(), "loo_min": float(loo.min()), "loo_mean": float(loo.mean()),
            "threshold": float(loo.min() - margin)}


def check_embed(score, threshold, simple_bg, relax):
    t = threshold if simple_bg else threshold - relax
    return {"passed": score >= t, "score": score,
            "detail": f"similarity {score:.3f} vs threshold {t:.3f}" + ("" if simple_bg else " (scene, relaxed)")}


# ---------------------------------------------------------------- vision-language model

VLM_SYSTEM = (
    "You are a strict visual QA inspector for character artwork. Judge only what is actually visible "
    "in the image, never what would be typical for the character. Answer each question in one to three "
    "words. If the feature cannot be seen, answer exactly \"not visible\". Reply with a single JSON object "
    "mapping each question id to its answer and nothing else."
)


class VLMJudge:
    def __init__(self, cfg):
        self.cfg = cfg

    def ask(self, image, questions):
        img = image.convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        text = "Questions:\n" + "\n".join(f"{q['id']}: {q['question']}" for q in questions)
        body = {
            "model": self.cfg["model"],
            "temperature": 0,
            "messages": [
                {"role": "system", "content": VLM_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": text},
                ]},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {self.cfg['api_key']}"
        req = urllib.request.Request(self.cfg["base_url"].rstrip("/") + "/chat/completions",
                                     json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=self.cfg.get("timeout_seconds", 180)) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise ValueError(f"VLM did not return JSON: {content[:300]}")
        return {str(k): str(v) for k, v in json.loads(m.group(0)).items()}


def _has_word(text, words):
    return any(re.search(rf"(?<![a-z0-9]){re.escape(w.lower())}(?![a-z0-9])", text) for w in words)


def grade_vlm(questions, answers, min_pass):
    results, crit_fail, noncrit = [], [], []
    for q in questions:
        a = answers.get(q["id"], "").strip().lower()
        if not a:
            ok, why = False, "no answer"
        elif "not visible" in a:
            ok = q.get("not_visible", "pass") == "pass"
            why = "not visible"
        elif _has_word(a, q.get("reject", [])):
            ok, why = False, f"'{a}' hits reject list"
        elif _has_word(a, q["expect"]):
            ok, why = True, a
        else:
            ok, why = False, f"'{a}' not in {q['expect']}"
        results.append((q["id"], ok, why))
        if q.get("critical", True):
            if not ok:
                crit_fail.append(q["id"])
        else:
            noncrit.append(ok)
    ratio = sum(noncrit) / len(noncrit) if noncrit else 1.0
    passed = not crit_fail and ratio >= min_pass
    fails = [f"{qid}: {why}" for qid, ok, why in results if not ok]
    return {"passed": passed, "score": sum(ok for _, ok, _ in results) / max(1, len(results)),
            "detail": "; ".join(fails) or "all checklist items pass", "answers": answers}


def check_vlm(judge, prep, asset, char):
    questions = [q for q in char["vlm"] if applies(q, asset)]
    if not questions:
        return {"passed": None, "score": None, "detail": "skipped: no checklist for this outfit"}
    try:
        answers = judge.ask(prep.full, questions)
    except Exception as e:  # a broken VLM must fail closed, never wave images through
        return {"passed": False, "score": 0.0, "detail": f"VLM error: {e}"}
    return grade_vlm(questions, answers, char["verify"]["vlm_min_pass"])
