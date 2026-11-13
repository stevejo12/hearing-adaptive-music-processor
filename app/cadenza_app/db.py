"""SQLite history of processed songs (one row per upload)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "history.sqlite3"
SONGS_DIR = DATA_DIR / "songs"

_lock = threading.Lock()
COLUMNS = ("id", "created", "name", "status", "error", "instruments", "gains",
           "audiogram_l", "audiogram_r", "before_file", "rating", "comment")


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _lock, _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            instruments TEXT NOT NULL,
            gains TEXT NOT NULL,
            audiogram_l TEXT NOT NULL,
            audiogram_r TEXT NOT NULL,
            before_file TEXT NOT NULL,
            rating INTEGER NOT NULL DEFAULT 0,
            comment TEXT NOT NULL DEFAULT '')""")
        existing = {r["name"] for r in c.execute("PRAGMA table_info(songs)")}
        for col, ddl in (("model_id", "TEXT NOT NULL DEFAULT 'baseline'"),
                         ("model_name", "TEXT NOT NULL DEFAULT 'Cadenza baseline (pretrained)'"),
                         ("residual", "REAL NOT NULL DEFAULT 0.1")):
            if col not in existing:                      # upgrade a history created by an older version
                c.execute(f"ALTER TABLE songs ADD COLUMN {col} {ddl}")
        c.execute("UPDATE songs SET status='error', error='Interrupted by a restart' "
                  "WHERE status='processing'")


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for key in ("instruments", "gains", "audiogram_l", "audiogram_r"):
        d[key] = json.loads(d[key])
    return d


def create(name: str, instruments, gains, levels_l, levels_r, before_ext: str,
           model_id: str = "baseline", model_name: str = "Cadenza baseline (pretrained)",
           residual: float = 0.1) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO songs (created, name, status, instruments, gains, audiogram_l, "
            "audiogram_r, before_file, model_id, model_name, residual) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), name, "processing",
             json.dumps(list(instruments)), json.dumps(list(gains)),
             json.dumps(list(levels_l)), json.dumps(list(levels_r)), f"before{before_ext}",
             model_id, model_name, residual))
        return cur.lastrowid


def get(song_id: int) -> dict | None:
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
    return _row(r) if r else None


def list_all(model_id: str | None = None) -> list[dict]:
    """All songs, newest first; only one model's songs if `model_id` is given."""
    with _lock, _conn() as c:
        if model_id is None:
            rows = c.execute("SELECT * FROM songs ORDER BY id DESC").fetchall()
        else:
            rows = c.execute("SELECT * FROM songs WHERE model_id=? ORDER BY id DESC", (model_id,)).fetchall()
    return [_row(r) for r in rows]


def set_status(song_id: int, status: str, error: str | None = None) -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE songs SET status=?, error=? WHERE id=?", (status, error, song_id))


def update_fields(song_id: int, rating: int | None = None, comment: str | None = None) -> bool:
    with _lock, _conn() as c:
        if rating is not None:
            c.execute("UPDATE songs SET rating=? WHERE id=?", (rating, song_id))
        if comment is not None:
            c.execute("UPDATE songs SET comment=? WHERE id=?", (comment, song_id))
        return c.execute("SELECT 1 FROM songs WHERE id=?", (song_id,)).fetchone() is not None


def count_by_model() -> dict[str, int]:
    with _lock, _conn() as c:
        return {r["model_id"]: r["n"] for r in
                c.execute("SELECT model_id, COUNT(*) AS n FROM songs GROUP BY model_id")}


def delete_by_model(model_id: str) -> list[int]:
    """Remove every song made with a model; returns their ids (to delete the audio files)."""
    with _lock, _conn() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM songs WHERE model_id=?", (model_id,))]
        c.execute("DELETE FROM songs WHERE model_id=?", (model_id,))
    return ids


def model_in_use(model_id: str) -> bool:
    """True if a song that is still processing uses this model."""
    with _lock, _conn() as c:
        return c.execute("SELECT 1 FROM songs WHERE model_id=? AND status='processing'",
                         (model_id,)).fetchone() is not None


def delete(song_id: int) -> bool:
    with _lock, _conn() as c:
        return c.execute("DELETE FROM songs WHERE id=?", (song_id,)).rowcount > 0
