"""Batch-generate Ren'Py character assets through ComfyUI with consistency gates.

  python pipeline.py lint                 pre-generation checks (bibles, asset rows, workflows, server models)
  python pipeline.py calibrate CHAR       derive verification thresholds from curated reference images
  python pipeline.py run [--dry-run]      generate -> verify -> retry failures -> approve best candidate
  python pipeline.py verify               re-verify existing candidates (after changing thresholds/checks)
  python pipeline.py approve ID FILE      manually approve a candidate that needs review
  python pipeline.py report               summary, report.csv and per-character contact sheets
  python pipeline.py export DEST          approved PNGs -> webp for the Ren'Py game folder
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import checks
import spec
from comfy import Comfy, ComfyError, patch_workflow

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- state

class State:
    """out/state.json: per-asset status (pending / approved / needs_review / blocked).
    out/manifest.jsonl: one line per generated candidate with seed, prompts and check results."""

    def __init__(self, out):
        self.out = out
        self.path = out / "state.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}

    def get(self, aid):
        return self.data.setdefault(aid, {"status": "pending", "attempts": 0})

    def save(self):
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def log(self, record):
        with open(self.out / "manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def candidates(self):
        recs = {}
        p = self.out / "manifest.jsonl"
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                recs[r["file"]] = r  # later verifications of the same file win
        return recs


def seed_for(char_id, asset_id, attempt, i):
    """Deterministic: re-running reproduces the same images, and retries never reuse a seed."""
    return int.from_bytes(hashlib.sha256(f"{char_id}|{asset_id}|{attempt}|{i}".encode()).digest()[:6], "big")


# ---------------------------------------------------------------- verification

class Verifier:
    def __init__(self, cfg, chars, short_circuit=True):
        self.cfg, self.chars = cfg, chars
        self.short_circuit = short_circuit
        self._embedder = None
        self._refs = {}
        self._judge = checks.VLMJudge(cfg["vlm"]) if cfg["vlm"].get("enabled", True) else None

    @property
    def embedder(self):
        if self._embedder is None:
            print("  loading embedding model", self.cfg["embed"]["model"])
            self._embedder = checks.Embedder(self.cfg["embed"]["model"], self.cfg["embed"].get("device", "auto"))
        return self._embedder

    def refs(self, char_id):
        if char_id not in self._refs:
            d = ROOT / self.cfg["refs_dir"] / char_id
            cal = json.loads(spec.calibration_path(self.cfg, char_id).read_text(encoding="utf-8"))
            self._refs[char_id] = (np.load(d / "ref_embeddings.npy"), cal)
        return self._refs[char_id]

    def verify(self, asset, path):
        char = self.chars[asset.character]
        v = char["verify"]
        prep = checks.Prepared(path, v["bg_delta_e"], asset.simple_bg)
        res = {}
        enabled = [c for c in spec.CHECK_NAMES if c in asset.checks]
        for name in enabled:
            if self.short_circuit and any(r["passed"] is False for r in res.values()):
                res[name] = {"passed": None, "score": None, "detail": "not run (earlier check failed)"}
                continue
            if name == "tech":
                res[name] = checks.check_tech(prep, asset, char)
            elif name == "palette":
                res[name] = checks.check_palette(prep, asset, char)
            elif name == "embed":
                ref_embs, cal = self.refs(asset.character)
                emb = self.embedder([prep.subject_crop()])[0]
                score = checks.topk_similarity(emb, ref_embs, cal["top_k"])
                res[name] = checks.check_embed(score, cal["threshold"], prep.simple_bg, v["scene_embed_relax"])
            elif name == "vlm":
                if self._judge is None:
                    res[name] = {"passed": None, "score": None, "detail": "skipped: vlm disabled in pipeline.toml"}
                else:
                    res[name] = checks.check_vlm(self._judge, prep, asset, char)
        passed = all(r["passed"] is not False for r in res.values()) and \
            ("embed" not in res or res["embed"]["passed"] is True) and res.get("tech", {}).get("passed", True)
        rank = next((res[c]["score"] for c in ("embed", "vlm", "palette") if res.get(c, {}).get("score") is not None), 0.0)
        return {"passed": passed, "rank": rank, "checks": res}


def fail_summary(result):
    return "; ".join(f"{k}: {r['detail']}" for k, r in result["checks"].items() if r["passed"] is False)


def decide(state, cfg, asset, verified, max_attempts):
    """Pick the best passing candidate; otherwise retry until attempts are exhausted."""
    st = state.get(asset.id)
    passing = [c for c in verified if c["result"]["passed"]]
    if passing:
        best = max(passing, key=lambda c: c["result"]["rank"])
        dest = ROOT / cfg["out_dir"] / "approved" / asset.character / (asset.id + Path(best["file"]).suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["file"], dest)
        st.update(status="approved", approved=str(dest.relative_to(ROOT)), candidate=best["file"],
                  seed=best.get("seed"), rank=best["result"]["rank"], reason="")
        print(f"  [OK]   {asset.id}: approved {Path(best['file']).name} (score {best['result']['rank']:.3f})")
        return True
    best = max(verified, key=lambda c: c["result"]["rank"]) if verified else None
    reason = fail_summary(best["result"]) if best else "no candidates"
    if st["attempts"] >= max_attempts:
        st.update(status="needs_review", candidate=best["file"] if best else None, reason=reason)
        print(f"  [FAIL] {asset.id}: needs review after {st['attempts']} attempts. Closest: {reason}")
        return True
    print(f"  [RETRY] {asset.id} (attempt {st['attempts']}, limit {max_attempts}): {reason}")
    return False


# ---------------------------------------------------------------- commands

def load(args):
    cfg = spec.load_config(ROOT)
    chars = spec.load_characters(cfg)
    assets = spec.load_assets(cfg)
    if getattr(args, "only", None):
        wanted = set(args.only.split(","))
        selected = [a for a in assets if a.id in wanted]
    else:
        selected = assets
    if getattr(args, "character", None):
        selected = [a for a in selected if a.character == args.character]
    return cfg, chars, assets, selected


def run_lint(cfg, chars, assets, online=True, quiet=False):
    lint = spec.Lint()
    spec.lint_assets(cfg, chars, assets, lint)
    if online:
        try:
            spec.lint_server(Comfy(cfg["comfy_url"]), chars, assets, lint)
        except ComfyError as e:
            lint.warn("server", f"model checks skipped: {e}")
    if not quiet or lint.errors:
        for level, where, msg in lint.items:
            if level == "ERROR" or not quiet:
                print(f"{level:5} {where}: {msg}")
    return lint


def cmd_lint(args):
    cfg, chars, assets, _ = load(args)
    lint = run_lint(cfg, chars, assets, online=not args.offline)
    print(f"\n{len(assets)} assets, {len(chars)} characters: {len(lint.errors)} errors, "
          f"{len(lint.items) - len(lint.errors)} warnings")
    if args.show_prompts:
        for a in assets:
            if a.character in chars:
                pos, neg = spec.build_prompts(chars[a.character], a)
                print(f"\n[{a.id}] ({a.character}, {a.workflow})\n  + {pos}\n  - {neg}")
    return 1 if lint.errors else 0


def cmd_calibrate(args):
    cfg = spec.load_config(ROOT)
    chars = spec.load_characters(cfg)
    char = chars.get(args.character)
    if not char:
        sys.exit(f"unknown character {args.character}")
    refs = spec.char_refs(cfg, char["id"])
    if len(refs) < 4:
        sys.exit(f"need >= 4 curated reference images in {cfg['refs_dir']}/{char['id']}/ (found {len(refs)})")
    v = char["verify"]
    preps = [checks.Prepared(r, v["bg_delta_e"], True) for r in refs]
    print(f"{len(refs)} references for {char['id']}")
    for p in preps:
        frac = p.mask.mean()
        if not v["min_subject_fraction"] <= frac <= v["max_subject_fraction"]:
            print(f"  WARN {p.path.name}: subject mask covers {frac:.0%}; references should be on a flat background")

    # Embedding threshold from the references themselves.
    emb = checks.Embedder(cfg["embed"]["model"], cfg["embed"].get("device", "auto"))
    ref_embs = emb([p.subject_crop() for p in preps])
    top_k = int(cfg["embed"].get("top_k", 3))
    cal = checks.calibrate_embed(ref_embs, top_k, v["embed_margin"])
    loo = np.array(cal["loo"])
    print(f"\nembedding: leave-one-out similarity min {loo.min():.3f} / mean {loo.mean():.3f} "
          f"-> threshold {cal['threshold']:.3f}")
    for r, s in zip(refs, loo):
        if s < loo.mean() - 2 * loo.std():
            print(f"  WARN {r.name} ({s:.3f}) looks unlike the rest of the set and drags the threshold down; consider removing it")
    d = ROOT / cfg["refs_dir"] / char["id"]
    np.save(d / "ref_embeddings.npy", ref_embs)
    spec.calibration_path(cfg, char["id"]).write_text(json.dumps(
        {"model": cfg["embed"]["model"], "refs": [r.name for r in refs], "top_k": top_k, **cal}, indent=1),
        encoding="utf-8")

    # Palette: the bible must accept its own references.
    if char["palette"]:
        print("\npalette coverage on references (min over refs / bible min_fraction):")
        all_fracs = [checks.palette_fractions(p, char["palette"])[0] for p in preps]
        for i, e in enumerate(char["palette"]):
            col = [f[i] for f in all_fracs]
            lo = e.get("min_fraction", 0.0)
            flag = ""
            if min(col) < lo:
                flag = f"  <-- rejects its own references; try hex/tolerance, or min_fraction ~{0.7 * min(col):.3f}"
            elif lo == 0 and e.get("max_fraction", 1) >= 1:
                flag = f"  (suggest min_fraction ~{0.7 * min(col):.3f})"
            print(f"  {e['name']:<16} {e['hex']}  min {min(col):6.1%} mean {np.mean(col):6.1%}  bible {lo:.1%}{flag}")
    print("\ndominant subject colours across references (for [[palette]] entries):")
    for hx, share in checks.suggest_palette(preps):
        print(f"  {hx}  {share:5.1%}")

    # VLM: every checklist item must pass on the references, otherwise the question or model is unreliable.
    if char["vlm"] and not args.skip_vlm and cfg["vlm"].get("enabled", True):
        print("\nVLM checklist on references:")
        judge = checks.VLMJudge(cfg["vlm"])
        bad = {}
        for p in preps:
            r = checks.grade_vlm(char["vlm"], judge.ask(p.full, char["vlm"]), 1.0)
            for item in r["detail"].split("; ") if not r["passed"] else []:
                bad.setdefault(item.split(":")[0], []).append(f"{p.path.name} -> {item}")
        if bad:
            for qid, lines in bad.items():
                print(f"  UNRELIABLE {qid}: fails on {len(lines)}/{len(preps)} references")
                for line in lines[:3]:
                    print("     ", line)
        else:
            print("  all checklist items pass on every reference")
    print(f"\nsaved {spec.calibration_path(cfg, char['id']).relative_to(ROOT)}")


def dependency_levels(selected, all_assets):
    by_id = {a.id: a for a in all_assets}
    level = {}

    def depth(a, seen=()):
        if a.id in level:
            return level[a.id]
        parent = by_id.get(a.init_image)
        level[a.id] = 0 if parent is None or a.id in seen else depth(parent, seen + (a.id,)) + 1
        return level[a.id]

    for a in selected:
        depth(a)
    out = {}
    for a in selected:
        out.setdefault(level[a.id], []).append(a)
    return [out[k] for k in sorted(out)]


def cmd_run(args):
    cfg, chars, assets, selected = load(args)
    lint = run_lint(cfg, chars, assets, online=not args.dry_run, quiet=True)
    if lint.errors:
        print(f"\nrefusing to generate: {len(lint.errors)} lint errors (run `python pipeline.py lint`)")
        return 1
    out = ROOT / cfg["out_dir"]
    state = State(out)
    todo = [a for a in selected if args.force or state.get(a.id)["status"] not in ("approved",)]
    if args.force:
        for a in todo:
            state.get(a.id)["status"] = "pending"
    if not todo:
        print("nothing to do (all selected assets approved; use --force to regenerate)")
        return 0

    if args.dry_run:
        for a in todo:
            char = chars[a.character]
            pos, neg = spec.build_prompts(char, a)
            n = a.candidates or char["verify"]["candidates"]
            att = state.get(a.id)["attempts"] + 1
            seeds = [seed_for(a.character, a.id, att, i) for i in range(n)]
            print(f"[{a.id}] {a.workflow} x{n} seeds {seeds}\n  + {pos}\n  - {neg}")
        return 0

    comfy = Comfy(cfg["comfy_url"], cfg.get("timeout_seconds", 900), cfg.get("poll_seconds", 1.0))
    verifier = Verifier(cfg, chars, short_circuit=not args.no_short_circuit)
    templates = {}
    # Attempt numbers keep counting across runs (so seeds never repeat), but each run gets a full retry budget.
    base_attempts = {a.id: state.get(a.id)["attempts"] for a in todo}
    t0 = time.time()

    for level in dependency_levels(todo, assets):
        pending = list(level)
        while pending:
            jobs = []
            for a in list(pending):
                char = chars[a.character]
                st = state.get(a.id)
                init = None
                if a.init_image:
                    parent = state.data.get(a.init_image)
                    if a.init_image.lower().endswith(spec.IMAGE_EXTS):
                        init = comfy.upload(ROOT / a.init_image)
                    elif parent and parent["status"] == "approved":
                        init = comfy.upload(ROOT / parent["approved"])
                    else:
                        st.update(status="blocked", reason=f"init_image '{a.init_image}' is not approved")
                        print(f"  [SKIP] {a.id}: {st['reason']}")
                        pending.remove(a)
                        continue
                ref = comfy.upload(ROOT / char["model"]["ref_image"]) if char["model"]["ref_image"] else None
                if a.workflow not in templates:
                    templates[a.workflow] = spec.load_workflow(cfg, a.workflow)
                st["attempts"] += 1
                pos, neg = spec.build_prompts(char, a)
                for i in range(a.candidates or char["verify"]["candidates"]):
                    seed = seed_for(a.character, a.id, st["attempts"], i)
                    prefix = f"apl/{a.character}/{a.id}/a{st['attempts']:02d}_c{i}"
                    wf, save_id = patch_workflow(templates[a.workflow], char, pos, neg, seed, prefix,
                                                 a.denoise, init, ref)
                    jobs.append({"asset": a, "seed": seed, "attempt": st["attempts"], "i": i,
                                 "save": save_id, "prompt_id": comfy.submit(wf), "positive": pos, "negative": neg})
            state.save()
            if not jobs:
                break
            print(f"\nqueued {len(jobs)} images for {len(pending)} assets; waiting for ComfyUI...")
            history = comfy.wait([j["prompt_id"] for j in jobs])
            if cfg.get("free_vram_before_verify", True):
                comfy.free()

            by_asset = {}
            for j in jobs:
                a = j["asset"]
                h = history[j["prompt_id"]]
                if h.get("_error"):
                    print(f"  [ERR]  {a.id} seed {j['seed']}: {h['_error'][:300]}")
                    continue
                images = h.get("outputs", {}).get(j["save"], {}).get("images", [])
                for k, img in enumerate(images[:1]):
                    d = out / "candidates" / a.character / a.id
                    d.mkdir(parents=True, exist_ok=True)
                    path = d / f"a{j['attempt']:02d}_c{j['i']}_{j['seed']}{Path(img['filename']).suffix}"
                    path.write_bytes(comfy.fetch(img))
                    result = verifier.verify(a, path)
                    rec = {"asset": a.id, "character": a.character, "file": str(path), "seed": j["seed"],
                           "attempt": j["attempt"], "workflow": a.workflow, "positive": j["positive"],
                           "negative": j["negative"], "result": result, "time": time.time()}
                    state.log(rec)
                    by_asset.setdefault(a.id, []).append(rec)

            for a in list(pending):
                char = chars[a.character]
                if decide(state, cfg, a, by_asset.get(a.id, []), base_attempts[a.id] + char["verify"]["max_attempts"]):
                    pending.remove(a)
            state.save()

    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min")
    write_report(cfg, chars, assets, state)
    return 0


def cmd_verify(args):
    """Re-run verification on every stored candidate, e.g. after recalibrating or editing the bible."""
    cfg, chars, assets, selected = load(args)
    state = State(ROOT / cfg["out_dir"])
    verifier = Verifier(cfg, chars, short_circuit=not args.no_short_circuit)
    recs = state.candidates()
    for a in selected:
        files = [r for r in recs.values() if r["asset"] == a.id and Path(r["file"]).is_file()]
        if not files:
            continue
        verified = []
        for r in files:
            r = {**r, "result": verifier.verify(a, Path(r["file"])), "time": time.time()}
            state.log(r)
            verified.append(r)
        st = state.get(a.id)
        decide(state, cfg, a, verified, max_attempts=st["attempts"])  # no retries here
    state.save()
    write_report(cfg, chars, assets, state)


def cmd_approve(args):
    cfg = spec.load_config(ROOT)
    state = State(ROOT / cfg["out_dir"])
    assets = {a.id: a for a in spec.load_assets(cfg)}
    a = assets.get(args.id)
    src = Path(args.file).resolve()
    if not a or not src.is_file():
        sys.exit("unknown asset id or file")
    dest = ROOT / cfg["out_dir"] / "approved" / a.character / (a.id + src.suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    state.get(a.id).update(status="approved", approved=str(dest.relative_to(ROOT)), candidate=str(src),
                           reason="manually approved")
    state.save()
    print(f"approved {a.id} <- {src.name}")


def contact_sheet(paths, labels, dest, thumb=256, cols=8):
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for n, (p, label) in enumerate(zip(paths, labels)):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = (n % cols) * thumb, (n // cols) * (thumb + 18)
        sheet.paste(im, (x + (thumb - im.size[0]) // 2, y))
        draw.text((x + 4, y + thumb + 3), label[:36], fill=(220, 220, 220))
    sheet.save(dest, quality=88)


def write_report(cfg, chars, assets, state):
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(out / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "character", "status", "attempts", "score", "approved", "closest_candidate", "reason"])
        for a in assets:
            st = state.data.get(a.id, {"status": "pending", "attempts": 0})
            counts[st["status"]] = counts.get(st["status"], 0) + 1
            w.writerow([a.id, a.character, st["status"], st["attempts"], st.get("rank", ""),
                        st.get("approved", ""), st.get("candidate", ""), st.get("reason", "")])
    for cid in chars:
        done = [(ROOT / state.data[a.id]["approved"], a.id) for a in assets
                if a.character == cid and state.data.get(a.id, {}).get("status") == "approved"]
        if done:
            contact_sheet([p for p, _ in done], [l for _, l in done], out / f"contact_{cid}.jpg")
    print("status: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for a in assets:
        st = state.data.get(a.id, {})
        if st.get("status") in ("needs_review", "blocked"):
            print(f"  {st['status']:12} {a.id}: {st.get('reason', '')}")
            if st.get("candidate"):
                print(f"               closest: {st['candidate']}")
    print(f"report: {(out / 'report.csv').relative_to(ROOT)}; contact sheets: {cfg['out_dir']}/contact_<character>.jpg")


def cmd_report(args):
    cfg, chars, assets, _ = load(args)
    write_report(cfg, chars, assets, State(ROOT / cfg["out_dir"]))


def cmd_export(args):
    cfg, chars, assets, selected = load(args)
    state = State(ROOT / cfg["out_dir"])
    dest = Path(args.dest)
    n = 0
    for a in selected:
        st = state.data.get(a.id, {})
        if st.get("status") != "approved":
            continue
        im = Image.open(ROOT / st["approved"])
        target = dest / a.character / f"{a.id}.webp"
        target.parent.mkdir(parents=True, exist_ok=True)
        im.save(target, "WEBP", quality=args.quality, method=6, lossless=args.lossless)
        n += 1
    print(f"exported {n} images to {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lint")
    p.add_argument("--offline", action="store_true", help="skip checking models on the ComfyUI server")
    p.add_argument("--show-prompts", action="store_true", help="print the assembled prompt for every asset")
    p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("calibrate")
    p.add_argument("character")
    p.add_argument("--skip-vlm", action="store_true")
    p.set_defaults(fn=cmd_calibrate)

    for name, fn in (("run", cmd_run), ("verify", cmd_verify), ("report", cmd_report), ("export", cmd_export)):
        p = sub.add_parser(name)
        p.add_argument("--only", help="comma-separated asset ids")
        p.add_argument("--character")
        if name == "run":
            p.add_argument("--force", action="store_true", help="regenerate even if approved")
            p.add_argument("--dry-run", action="store_true", help="print prompts and seeds, generate nothing")
        if name in ("run", "verify"):
            p.add_argument("--no-short-circuit", action="store_true",
                           help="run every check even after one fails (slower, better diagnostics)")
        if name == "export":
            p.add_argument("dest")
            p.add_argument("--quality", type=int, default=92)
            p.add_argument("--lossless", action="store_true")
        p.set_defaults(fn=fn)

    p = sub.add_parser("approve")
    p.add_argument("id")
    p.add_argument("file")
    p.set_defaults(fn=cmd_approve)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
