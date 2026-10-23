"""Minimal ComfyUI HTTP client (stdlib only) plus workflow patching."""
import copy
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from spec import titled_nodes


class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, url, timeout=900, poll=1.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.poll = poll
        self.client_id = uuid.uuid4().hex
        self._uploads = {}

    def _request(self, path, data=None, headers=None, raw=False):
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(f"{path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:2000]}") from None
        except urllib.error.URLError as e:
            raise ComfyError(f"cannot reach ComfyUI at {self.url} ({e.reason})") from None
        return body if raw else json.loads(body)

    def choices(self, node_class, input_name):
        info = self._request(f"/object_info/{node_class}")[node_class]["input"]
        spec = {**info.get("required", {}), **info.get("optional", {})}[input_name]
        # Older servers: [[choices...], {...}]; newer: ["COMBO", {"options": [...]}]
        if isinstance(spec[0], list):
            return spec[0]
        return spec[1].get("options", [])

    def upload(self, path):
        """Upload a local image once per run; returns the name LoadImage expects."""
        path = Path(path).resolve()
        key = (str(path), path.stat().st_mtime)
        if key in self._uploads:
            return self._uploads[key]
        boundary = uuid.uuid4().hex
        # Prefix with a short hash so different files with the same basename don't collide.
        name = f"apl_{uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex[:8]}_{path.name}"
        body = b"".join([
            f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode(),
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        r = self._request("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        result = f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"]
        self._uploads[key] = result
        return result

    def submit(self, workflow):
        r = self._request("/prompt", json.dumps({"prompt": workflow, "client_id": self.client_id}).encode(),
                          {"Content-Type": "application/json"})
        if r.get("node_errors"):
            raise ComfyError(f"workflow rejected: {json.dumps(r['node_errors'])[:2000]}")
        return r["prompt_id"]

    def wait(self, prompt_ids):
        """Block until every prompt finished; returns {prompt_id: history entry}."""
        pending, done = set(prompt_ids), {}
        deadline = time.monotonic() + self.timeout
        while pending:
            for pid in list(pending):
                h = self._request(f"/history/{pid}")
                if pid in h and h[pid].get("status", {}).get("completed") is not None:
                    status = h[pid]["status"]
                    if status.get("status_str") == "error" or not status.get("completed"):
                        msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                        h[pid]["_error"] = json.dumps(msgs)[:2000] or "execution failed"
                    done[pid] = h[pid]
                    pending.discard(pid)
            if pending:
                if time.monotonic() > deadline:
                    raise ComfyError(f"timed out waiting for {len(pending)} prompt(s)")
                time.sleep(self.poll)
        return done

    def fetch(self, image):
        q = urllib.parse.urlencode({"filename": image["filename"], "subfolder": image.get("subfolder", ""),
                                    "type": image.get("type", "output")})
        return self._request(f"/view?{q}", raw=True)

    def free(self):
        """Ask ComfyUI to unload models so the verifier (embedder/VLM) has VRAM."""
        try:
            self._request("/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
                          {"Content-Type": "application/json"}, raw=True)
        except ComfyError:
            pass


def patch_workflow(template, char, positive, negative, seed, prefix, denoise=None,
                   init_image=None, ref_image=None):
    """Fill the titled nodes of an API-format workflow. Every generation parameter
    comes from the character bible, so no two assets of a character can differ in
    model, LoRA strength, sampler, CFG, steps or resolution."""
    wf = copy.deepcopy(template)
    t = titled_nodes(wf)
    m = char["model"]

    def node(title):
        return wf[t[title]]["inputs"]

    node("CHECKPOINT")["ckpt_name"] = m["checkpoint"]
    if "CLIP_SKIP" in t:
        node("CLIP_SKIP")["stop_at_clip_layer"] = -abs(int(m["clip_skip"]))
    lora_titles = sorted(x for x in t if x.startswith("LORA_"))
    for i, title in enumerate(lora_titles):
        lora = m["loras"][i] if i < len(m["loras"]) else {**m["loras"][0], "strength": 0.0}
        n = node(title)
        n["lora_name"] = lora["name"]
        n["strength_model"] = float(lora.get("strength", 1.0))
        n["strength_clip"] = float(lora.get("strength_clip", lora.get("strength", 1.0)))
    node("POSITIVE")["text"] = positive
    node("NEGATIVE")["text"] = negative
    s = node("SAMPLER")
    s.update(seed=seed, steps=int(m["steps"]), cfg=float(m["cfg"]), sampler_name=m["sampler"], scheduler=m["scheduler"])
    if denoise is not None:
        s["denoise"] = denoise
    if "LATENT" in t:
        node("LATENT").update(width=int(m["width"]), height=int(m["height"]), batch_size=1)
    if "INIT_IMAGE" in t:
        node("INIT_IMAGE")["image"] = init_image
    if "REF_IMAGE" in t:
        node("REF_IMAGE")["image"] = ref_image
    node("SAVE")["filename_prefix"] = prefix
    return wf, t["SAVE"]
