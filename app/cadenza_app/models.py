"""Pretrained Cadenza models and user-uploaded model bundles.

* The built-in "baseline" models are checked at start-up and downloaded only if missing,
  into the persistent /data volume (HF_HOME=/data/hf). While that runs, `status()` tells the
  page to show a notice.
* Extra models are uploaded from the UI as a .zip "bundle" and stored in /data/models/<id>/,
  so they stay on the server for every user.

Bundle layout (a single wrapping folder is fine):

    my_model.zip
      Flute/config.json
      Flute/model.safetensors
      Oboe/config.json
      Oboe/model.safetensors      <- any subset of the supported instruments

Instruments missing from a bundle fall back to the baseline model. Only safetensors weights are
accepted: pickle-based files (.pt/.bin/.ckpt) can execute code when loaded, so they are rejected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from . import profile

logger = logging.getLogger(__name__)

BASELINE_ID = "baseline"
BASELINE_NAME = "Cadenza baseline (pretrained)"
CAUSALITY = os.environ.get("CADENZA_CAUSALITY", "noncausal")  # causal | noncausal
_SUFFIX = {"causal": "Causal", "noncausal": "NonCausal"}[CAUSALITY]
MODEL_FILES = ("config.json", "model.safetensors")
PICKLE_SUFFIXES = {".pt", ".pth", ".bin", ".pkl", ".pickle", ".ckpt"}
_ID_RE = re.compile(r"^[0-9a-f]{8}$")

_lock = threading.Lock()
_state = {"state": "checking", "message": "Checking model files...", "done": 0,
          "total": 0, "error": None}


# ---------------------------------------------------------------- baseline download
def repos() -> dict[str, str]:
    return {name: f"cadenzachallenge/ConvTasNet_{name}_{_SUFFIX}" for name in profile.INSTRUMENT_CENTRE_HZ}


def status() -> dict:
    with _lock:
        return dict(_state)


def is_ready() -> bool:
    return status()["state"] == "ready"


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def local_path(repo: str) -> str:
    """Local snapshot folder of a baseline model that is already downloaded (no network)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True)


def ensure_models(fetch=None) -> None:
    """Make sure every baseline model is on disk; download the missing ones. `fetch` is for tests."""
    if fetch is None:
        from huggingface_hub import snapshot_download as fetch  # type: ignore[assignment]
    all_repos = repos()
    _set(state="checking", message="Checking model files...", done=0, total=len(all_repos), error=None)
    try:
        missing = {}
        for name, repo in all_repos.items():
            try:
                fetch(repo, local_files_only=True)
            except Exception:  # not cached yet
                missing[name] = repo
        done = len(all_repos) - len(missing)
        if missing:
            logger.info("Downloading %d model(s): %s", len(missing), ", ".join(missing))
            for name, repo in missing.items():
                _set(state="downloading", done=done,
                     message=f"Downloading the pretrained Cadenza models ({done + 1} of {len(all_repos)}: "
                             f"{name}). Only needed on the first start; please wait.")
                fetch(repo)
                done += 1
        _set(state="ready", message="Models ready.", done=len(all_repos))
    except Exception as exc:
        logger.exception("Model download failed")
        _set(state="error", error=str(exc)[:300],
             message="Could not download the Cadenza models (is the server online?). "
                     "Restart the container to retry.")


def start_in_background() -> threading.Thread:
    t = threading.Thread(target=ensure_models, name="model-download", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------- uploaded bundles
def _root() -> Path:
    return Path(os.environ.get("DATA_DIR", "/data")) / "models"


def valid_id(model_id: str) -> bool:
    return model_id == BASELINE_ID or bool(_ID_RE.match(model_id))


def _read_meta(model_id: str) -> Optional[dict]:
    try:
        return json.loads((_root() / model_id / "meta.json").read_text())
    except (OSError, ValueError):
        return None


def list_models() -> list[dict]:
    """Baseline first, then uploaded models, oldest first."""
    out = [{"id": BASELINE_ID, "name": BASELINE_NAME, "builtin": True, "created": None,
            "instruments": list(profile.INSTRUMENT_CENTRE_HZ)}]
    root = _root()
    found = []
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and _ID_RE.match(d.name):
                meta = _read_meta(d.name)
                if meta:
                    found.append({**meta, "id": d.name, "builtin": False})
    return out + sorted(found, key=lambda m: m.get("created", ""))


def get_model(model_id: str) -> Optional[dict]:
    return next((m for m in list_models() if m["id"] == model_id), None)


def model_paths(model_id: str) -> dict[str, str]:
    """Folder of every instrument's weights for a model (baseline fills any gaps)."""
    paths = {name: local_path(repo) for name, repo in repos().items()}
    if model_id != BASELINE_ID:
        meta = get_model(model_id)
        if meta is None:
            raise ValueError("Unknown model")
        for inst in meta["instruments"]:
            paths[inst] = str(_root() / model_id / inst)
    return paths


def delete_model(model_id: str) -> bool:
    if model_id == BASELINE_ID or not valid_id(model_id):
        return False
    target = _root() / model_id
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def _clean_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    name = info.filename
    if "\\" in name or name.startswith("/"):
        raise ValueError("The zip contains unsafe file paths")
    parts = PurePosixPath(name).parts
    if any(p in ("..", "") for p in parts):
        raise ValueError("The zip contains unsafe file paths")
    return parts


def register_bundle(zip_path: Path, name: str, max_bytes: int,
                    check: Optional[Callable[[str], None]] = None) -> dict:
    """Validate an uploaded bundle and store it. Raises ValueError with a readable message.

    `check(path)` is called for every instrument folder to prove the weights really load.
    """
    name = " ".join(name.split())[:60] or "Uploaded model"
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Upload a .zip file (see the layout in the page help)")
    with zipfile.ZipFile(zip_path) as zf:
        files = [(i, _clean_parts(i)) for i in zf.infolist() if not i.is_dir()]
        files = [(i, p) for i, p in files if p[0] != "__MACOSX" and p[-1] != ".DS_Store"]
        for _, parts in files:
            if PurePosixPath(parts[-1]).suffix.lower() in PICKLE_SUFFIXES:
                raise ValueError("Pickle-based weights (.pt/.bin/.ckpt) are not accepted; "
                                 "export model.safetensors")
        tops = {p[0] for _, p in files}
        if files and len(tops) == 1 and not tops & set(profile.INSTRUMENT_CENTRE_HZ) \
                and all(len(p) >= 2 for _, p in files):
            files = [(i, p[1:]) for i, p in files]          # strip one wrapping folder
        wanted = {}
        for info, parts in files:
            if len(parts) == 2 and parts[0] in profile.INSTRUMENT_CENTRE_HZ and parts[1] in MODEL_FILES:
                wanted[(parts[0], parts[1])] = info
        instruments = [i for i in profile.INSTRUMENT_CENTRE_HZ
                       if all((i, f) in wanted for f in MODEL_FILES)]
        if not instruments:
            raise ValueError("No usable model found. Expected <Instrument>/config.json and "
                             "<Instrument>/model.safetensors for: " + ", ".join(profile.INSTRUMENT_CENTRE_HZ))
        if sum(wanted[(i, f)].file_size for i in instruments for f in MODEL_FILES) > max_bytes:
            raise ValueError("The model is larger than the allowed size")

        model_id = secrets.token_hex(4)
        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".tmp-{model_id}"
        try:
            written = 0
            for inst in instruments:
                (staging / inst).mkdir(parents=True, exist_ok=True)
                for fname in MODEL_FILES:
                    with zf.open(wanted[(inst, fname)]) as src, (staging / inst / fname).open("wb") as dst:
                        while chunk := src.read(1024 * 1024):
                            written += len(chunk)
                            if written > max_bytes:      # real size, not the declared one
                                raise ValueError("The model is larger than the allowed size")
                            dst.write(chunk)
                try:
                    json.loads((staging / inst / "config.json").read_text())
                except ValueError as exc:
                    raise ValueError(f"{inst}/config.json is not valid JSON") from exc
                if check is not None:
                    check(str(staging / inst))
            meta = {"name": name, "instruments": instruments, "size_mb": round(written / 1e6, 1),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (staging / "meta.json").write_text(json.dumps(meta))
            staging.rename(root / model_id)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return {**meta, "id": model_id, "builtin": False}
