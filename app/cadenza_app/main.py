"""Web app: login, upload song + hearing profile, history table with players."""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from . import db, models, pipeline, profile

logger = logging.getLogger("cadenza_app")
logging.basicConfig(level=logging.INFO)

APP_USER = os.environ.get("APP_USER", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "admin")


def _load_secret() -> str:
    """Session-signing key: APP_SECRET if set, else one generated once and kept in /data
    (so logins survive container restarts and rebuilds)."""
    if os.environ.get("APP_SECRET"):
        return os.environ["APP_SECRET"]
    path = db.DATA_DIR / "secret_key"
    try:
        if path.exists():
            return path.read_text().strip()
        db.DATA_DIR.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        path.write_text(key)
        path.chmod(0o600)
        return key
    except OSError:
        return secrets.token_hex(32)


SECRET = _load_secret()
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "50"))
MAX_MODEL_MB = float(os.environ.get("MAX_MODEL_MB", "2048"))   # uploaded model bundles
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
STATIC_DIR = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(_app):
    db.init()
    db.SONGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = db.DATA_DIR / "tmp"          # big uploads spool here (the volume), not the container layer
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    models.start_in_background()
    if APP_USER == "admin" and APP_PASSWORD == "admin":
        logger.warning("Using the default admin/admin login (set APP_USER/APP_PASSWORD)")
    yield


app = FastAPI(title="Cadenza hearing-adaptive music processor", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET, max_age=60 * 60 * 24 * 7,
                   same_site="lax", https_only=False)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_worker = ThreadPoolExecutor(max_workers=1)  # one song at a time: CPU/RAM bound


# ---------- auth ----------
def require_login(request: Request) -> None:
    if not request.session.get("user"):
        raise HTTPException(status_code=401, detail="Login required")


@app.get("/login", response_class=HTMLResponse)
def login_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...), request: Request = None):
    ok = secrets.compare_digest(username.encode(), APP_USER.encode()) and \
        secrets.compare_digest(password.encode(), APP_PASSWORD.encode())
    if not ok:
        return RedirectResponse("/login?error=1", status_code=303)
    request.session["user"] = username
    return RedirectResponse("/", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=303)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


# ---------- API ----------
def _song_json(s: dict) -> dict:
    done = s["status"] == "done"
    return {
        "id": s["id"], "created": s["created"], "name": s["name"], "status": s["status"],
        "error": s["error"], "instruments": s["instruments"], "gains": s["gains"],
        "rating": s["rating"], "comment": s["comment"],
        "model_id": s["model_id"], "model_name": s["model_name"], "residual": s["residual"],
        "audiogram_l": s["audiogram_l"], "audiogram_r": s["audiogram_r"],
        "before_url": f"/media/{s['id']}/before",
        "after_url": f"/media/{s['id']}/after" if done else None,
    }


@app.get("/api/config", dependencies=[Depends(require_login)])
def config() -> dict:
    return {"instruments": pipeline.INSTRUMENTS, "frequencies": [int(f) for f in profile.AUDIOGRAM_FREQS],
            "max_upload_mb": MAX_UPLOAD_MB, "max_seconds": pipeline.MAX_SECONDS,
            "max_model_mb": MAX_MODEL_MB, "default_residual": pipeline.DEFAULT_RESIDUAL}


@app.get("/api/status", dependencies=[Depends(require_login)])
def model_status() -> dict:
    return models.status()


@app.get("/api/songs", dependencies=[Depends(require_login)])
def list_songs(model: Optional[str] = None) -> list[dict]:
    """History table rows. `?model=<id>` shows only the songs made with that model."""
    return [_song_json(s) for s in db.list_all(model)]


def _parse_floats(text: str, what: str) -> list[float]:
    try:
        return [float(x) for x in text.replace(";", ",").split(",") if x.strip() != ""]
    except ValueError as exc:
        raise HTTPException(422, f"{what}: numbers only") from exc


def _save_upload(upload: UploadFile, dest: Path, max_mb: float = None) -> None:
    max_mb = MAX_UPLOAD_MB if max_mb is None else max_mb
    limit = int(max_mb * 1024 * 1024)
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File is larger than {max_mb:g} MB")
            out.write(chunk)


def _run_job(song_id: int, instruments, gains, levels_l, levels_r, model_id, residual) -> None:
    folder = db.SONGS_DIR / str(song_id)
    song = db.get(song_id)
    if song is None:  # deleted while queued
        return
    try:
        pipeline.process(folder / song["before_file"], folder / "after.mp3",
                         instruments, gains, np.array(levels_l), np.array(levels_r),
                         model_id, residual)
        if db.get(song_id) is not None:
            db.set_status(song_id, "done")
    except Exception as exc:  # shown in the row; full trace in the log
        logger.exception("Processing failed for song %s", song_id)
        if db.get(song_id) is not None:
            db.set_status(song_id, "error", str(exc)[:300])


@app.post("/api/process", dependencies=[Depends(require_login)])
def process_song(song: UploadFile = File(...), audiogram_l: str = Form(...),
                 audiogram_r: str = Form(...), instruments: str = Form(...),
                 gains: str = Form(""), model: str = Form(models.BASELINE_ID),
                 residual: float = Form(pipeline.DEFAULT_RESIDUAL)) -> dict:
    if not models.is_ready():
        raise HTTPException(503, models.status()["message"])
    ext = Path(song.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(422, "Upload an audio file (mp3, wav, flac, m4a, ogg)")
    try:
        levels_l = profile.parse_levels(_parse_floats(audiogram_l, "Left audiogram"))
        levels_r = profile.parse_levels(_parse_floats(audiogram_r, "Right audiogram"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    chosen = models.get_model(model) if models.valid_id(model) else None
    if chosen is None:
        raise HTTPException(422, "Unknown model")
    if not 0.0 <= residual <= 1.0:
        raise HTTPException(422, "Residual must be between 0 and 1")

    names = [i.strip() for i in instruments.split(",") if i.strip()]
    if not 2 <= len(names) <= 3:
        raise HTTPException(422, "Choose 2 or 3 instruments (woodwind trios only)")
    bad = [n for n in names if n not in pipeline.INSTRUMENTS]
    if bad:
        raise HTTPException(422, f"Unsupported instrument(s): {', '.join(bad)}")

    if gains.strip():
        gain_list = _parse_floats(gains, "Gains")
        if len(gain_list) != len(names) or any(abs(g) > 20 for g in gain_list):
            raise HTTPException(422, "Give one gain per instrument, within +/-20 dB")
    else:
        gain_list = profile.default_gains(names, levels_l, levels_r)

    song_id = db.create((song.filename or "song")[:120], names, gain_list,
                        levels_l.tolist(), levels_r.tolist(), ext,
                        chosen["id"], chosen["name"], residual)
    try:
        _save_upload(song, db.SONGS_DIR / str(song_id) / f"before{ext}")
    except HTTPException:
        db.delete(song_id)
        shutil.rmtree(db.SONGS_DIR / str(song_id), ignore_errors=True)
        raise
    _worker.submit(_run_job, song_id, names, gain_list, levels_l.tolist(), levels_r.tolist(),
                   chosen["id"], residual)
    return _song_json(db.get(song_id))


# ---------- models ----------
@app.get("/api/models", dependencies=[Depends(require_login)])
def list_models() -> list[dict]:
    counts = db.count_by_model()
    return [{**m, "songs": counts.get(m["id"], 0)} for m in models.list_models()]


@app.post("/api/models", dependencies=[Depends(require_login)])
def upload_model(file: UploadFile = File(...), name: str = Form("")) -> dict:
    """Add a model bundle (.zip, see models.py). Stays on the server for every user."""
    tmp = Path(tempfile.gettempdir()) / f"model-upload-{secrets.token_hex(6)}.zip"
    try:
        _save_upload(file, tmp, MAX_MODEL_MB)
        default_name = Path(file.filename or "").stem
        return models.register_bundle(tmp, name.strip() or default_name,
                                      int(MAX_MODEL_MB * 1024 * 1024), check=pipeline.check_loadable)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)


@app.delete("/api/models/{model_id}", dependencies=[Depends(require_login)])
def delete_model(model_id: str) -> dict:
    if model_id == models.BASELINE_ID:
        raise HTTPException(409, "The built-in baseline model cannot be deleted")
    if db.model_in_use(model_id):
        raise HTTPException(409, "A song is still being processed with this model")
    if not models.delete_model(model_id):
        raise HTTPException(404, "No such model")
    pipeline.forget_model(model_id)
    removed = db.delete_by_model(model_id)          # each model has its own table of songs
    for song_id in removed:
        shutil.rmtree(db.SONGS_DIR / str(song_id), ignore_errors=True)
    return {"deleted": model_id, "songs_deleted": len(removed)}


class Patch(BaseModel):
    rating: Optional[int] = Field(None, ge=0, le=5)
    comment: Optional[str] = Field(None, max_length=2000)


@app.patch("/api/songs/{song_id}", dependencies=[Depends(require_login)])
def patch_song(song_id: int, body: Patch) -> dict:
    if not db.update_fields(song_id, body.rating, body.comment):
        raise HTTPException(404, "No such song")
    return _song_json(db.get(song_id))


@app.delete("/api/songs/{song_id}", dependencies=[Depends(require_login)])
def delete_song(song_id: int) -> dict:
    if not db.delete(song_id):
        raise HTTPException(404, "No such song")
    shutil.rmtree(db.SONGS_DIR / str(song_id), ignore_errors=True)
    return {"deleted": song_id}


@app.get("/media/{song_id}/{which}", dependencies=[Depends(require_login)])
def media(song_id: int, which: str):
    song = db.get(song_id)
    if song is None or which not in ("before", "after"):
        raise HTTPException(404, "Not found")
    folder = db.SONGS_DIR / str(song_id)
    if which == "before":
        path = folder / song["before_file"]
    else:   # MP3 for the browser; older rows may still have the original FLAC
        path = next((p for p in (folder / "after.mp3", folder / "after.flac") if p.exists()), folder / "after.mp3")
    if not path.exists():
        raise HTTPException(404, "Not ready")
    return FileResponse(path)


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    if exc.status_code == 401 and not request.url.path.startswith("/api/") \
            and not request.url.path.startswith("/media/"):
        return RedirectResponse("/login", status_code=303)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
