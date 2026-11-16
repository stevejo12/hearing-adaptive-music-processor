from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import resample_poly

import soundfile as sf

from clarity.utils.audiogram import Listener

SAMPLE_RATE = 44100  # input_sample_rate in the baseline config.yaml


def read_flac(path: Path) -> tuple[np.ndarray, int]:
    signal, sr = sf.read(str(path), dtype="int16", always_2d=False)
    signal = (signal / 32768.0).astype(np.float32)
    scale_file = Path(path).with_suffix(".txt")
    if scale_file.exists():
        signal *= float(scale_file.read_text(encoding="utf-8"))
    return signal, sr

# Used to label where a track comes from (real recording vs synthetic render)
REAL_DATASETS = ("URMP", "BACH10", "Bach10")
SYNTHETIC_DATASETS = ("EnsembleSet", "CadenzaWoodwind")


class CAD2Loader:
    def __init__(
        self,
        root: str | Path = "/content/cadenza_data/cad2/task2",
        clarity_dir: str | Path = "/content/clarity",
    ):
        self.root = Path(root)
        self.meta = self.root / "metadata"
        self.audio = self.root / "audio"
        self.config_file = Path(clarity_dir) / "recipes/cad2/task2/baseline/config.yaml"

    # metadata
    @lru_cache(maxsize=None)
    def _json(self, name: str) -> dict:
        path = self.meta / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run notebook 00 for this split")
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _json_optional(self, name: str) -> dict:
        try:
            return self._json(name)
        except FileNotFoundError:
            return {}

    def song_dataset(self, split: str, song_name: str) -> str:
        tracks = [v["track"] for v in self.music(split)[song_name].values()]
        for t in tracks:
            for name in REAL_DATASETS + SYNTHETIC_DATASETS:
                if name in t:
                    return name
        if split == "valid":
            info = self._json_optional("music.valid.to_generate.json").get(song_name, {})
            found = set()
            for v in info.values():
                if isinstance(v, dict) and v.get("source_dataset"):
                    parts = [p for p in str(v["source_dataset"]).split("/") if p]  # skip empty parts
                    if parts:
                        found.add(parts[0])
            if found:
                return "+".join(sorted(found))
        if split == "train":
            for info in self._json_optional("music_tracks.train.json").values():
                mix = info.get("mixture_path")
                if mix and any(t.endswith(mix) for t in tracks):
                    return info.get("dataset", "unknown")
        return "unknown"

    def scenes(self, split: str) -> dict:
        return self._json(f"scenes.{split}.json")

    def music(self, split: str) -> dict:
        return self._json(f"music.{split}.json")

    def gains_table(self) -> dict:
        return self._json("gains.json")

    def list_pairs(self, split: str) -> list[tuple[str, str]]:
        scene_listeners = self._json(f"scene_listeners.{split}.json")
        return [(s, l) for s, ls in scene_listeners.items() for l in ls]

    # listeners
    @lru_cache(maxsize=None)
    def listeners(self, split: str) -> dict[str, Listener]:
        return Listener.load_listener_dict(self.meta / f"listeners.{split}.json")

    def compressor_params(self, split: str) -> dict:
        return self._json(f"compressor_params.{split}.json")

    @lru_cache(maxsize=None)
    def shared_compressor_settings(self) -> dict:
        with self.config_file.open(encoding="utf-8") as f:
            enhancer = yaml.safe_load(f)["enhancer"]
        return {
            "crossover_frequencies": list(enhancer["crossover_frequencies"]),
            "attack": list(enhancer["attack"]),
            "release": list(enhancer["release"]),
            "threshold": list(enhancer["threshold"]),
        }

    def get_compressor(self, listener_id: str, split: str) -> dict:
        shared = self.shared_compressor_settings()
        params = self.compressor_params(split)[listener_id]
        out = {"crossover_frequencies": shared["crossover_frequencies"]}
        for ear, side in (("left", "l"), ("right", "r")):
            out[ear] = {
                "attack": shared["attack"],
                "release": shared["release"],
                "threshold": shared["threshold"],
                "ratio": params[f"cr_{side}"],
                "makeup_gain": params[f"gain_{side}"],
            }
        return out

    def check_listener_match(self, split: str) -> dict:
        audiogram_ids = set(self._json(f"listeners.{split}.json"))
        compressor_ids = set(self.compressor_params(split))
        return {
            "n_listeners": len(audiogram_ids),
            "n_compressor": len(compressor_ids),
            "missing_compressor": sorted(audiogram_ids - compressor_ids),
            "missing_audiogram": sorted(compressor_ids - audiogram_ids),
        }

    # audio
    @staticmethod
    def to_standard(signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[:, None]
        if signal.shape[0] < signal.shape[1]:  # (channels, samples) -> (samples, channels)
            signal = signal.T
        if signal.shape[1] == 1:  # mono -> stereo
            signal = np.repeat(signal, 2, axis=1)
        return signal

    def _path(self, track: str, split: str | None = None) -> Path:
        path = self.audio / track
        if not path.exists() and split and (self.audio / split / track).exists():
            return self.audio / split / track
        return path

    def load_clip(self, entry: dict, cut: bool = True, split: str | None = None) -> np.ndarray:
        path = self._path(entry["track"], split)
        if not path.exists():
            tried = [str(path)] + ([str(self.audio / split / entry["track"])] if split else [])
            raise FileNotFoundError(f"{entry['track']} not found (tried {' and '.join(tried)})")
        signal, sr = read_flac(path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{entry['track']}: expected {SAMPLE_RATE} Hz, got {sr}")
        signal = self.to_standard(signal)
        n = int(round(entry["duration"] * sr))
        if cut and len(signal) > n + 1:
            start = int(round(entry["start"] * sr))
            signal = signal[start : start + n]
        return signal

    def _mixture_entry(self, song: dict) -> dict | None:
        if "mixture" in song:
            return song["mixture"]
        first = next(v for k, v in song.items() if k.startswith("source"))
        song_dir = self.audio / Path(*first["track"].split("/")[:3])
        mixes = sorted(song_dir.glob("mix_*.flac"))
        if not mixes:
            return None
        return {
            "track": str(mixes[0].relative_to(self.audio)),
            "start": first["start"],
            "duration": first["duration"],
            "instrument": "Mixture",
        }

    # scene
    def load_scene(
        self,
        scene_id: str,
        listener_id: str | None = None,
        split: str = "valid",
        target_sr: int | None = None,
        load_stems: bool = True,
    ) -> dict:
        scene = self.scenes(split)[scene_id]
        song = self.music(split)[scene["music"]]
        if listener_id is None:
            listener_id = self._json(f"scene_listeners.{split}.json")[scene_id][0]

        mix_entry = self._mixture_entry(song)
        if mix_entry is None:
            raise FileNotFoundError(f"no mixture found for {scene_id} ({scene['music']})")
        mixture = self.load_clip(mix_entry, split=split)

        sources = {k: v for k, v in song.items() if k.startswith("source")}
        instruments = {k: v["instrument"].split("_")[0] for k, v in sources.items()}
        stems, missing = {}, []
        if load_stems:
            for k, entry in sources.items():
                if self._path(entry["track"], split).exists():
                    stems[k] = self.load_clip(entry, split=split)
                else:
                    missing.append(k)

        n = min([len(mixture)] + [len(s) for s in stems.values()])
        mixture = mixture[:n]
        stems = {k: s[:n] for k, s in stems.items()}

        sr = SAMPLE_RATE
        if target_sr and target_sr != SAMPLE_RATE:
            g = np.gcd(SAMPLE_RATE, target_sr)
            up, down = target_sr // g, SAMPLE_RATE // g
            mixture = resample_poly(mixture, up, down, axis=0).astype(np.float32)
            stems = {k: resample_poly(s, up, down, axis=0).astype(np.float32) for k, s in stems.items()}
            sr = target_sr

        compressor = None
        if listener_id in self.compressor_params(split):
            compressor = self.get_compressor(listener_id, split)

        return {
            "scene_id": scene_id,
            "listener_id": listener_id,
            "split": split,
            "song": scene["music"],
            "sample_rate": sr,
            "mixture": mixture,
            "stems": stems,
            "missing_stems": missing,
            "instruments": instruments,
            "gains": self.gains_table()[scene["gain"]],
            "listener": self.listeners(split)[listener_id],
            "compressor": compressor,
            "dataset": (dataset := self.song_dataset(split, scene["music"])),
            "origin": track_origin(dataset),
        }

    # checks
    def check_scene(self, scene_id: str, split: str = "valid", tol_db: float = -30.0) -> dict:
        row = {"scene_id": scene_id, "split": split, "ok": False, "error": ""}
        try:
            s = self.load_scene(scene_id, split=split)
            arrays = [s["mixture"], *s["stems"].values()]
            row["seconds"] = len(s["mixture"]) / s["sample_rate"]
            row["n_sources"] = len(s["instruments"])
            row["missing_stems"] = len(s["missing_stems"])
            row["same_shape"] = len({a.shape for a in arrays}) == 1 and arrays[0].shape[1] == 2
            row["gains_match"] = set(s["gains"]) == set(s["instruments"])
            if s["stems"] and not s["missing_stems"]:
                err = np.sum((s["mixture"] - sum(s["stems"].values())) ** 2)
                row["mix_mismatch_db"] = float(10 * np.log10(err / (np.sum(s["mixture"] ** 2) + 1e-12) + 1e-12))
                row["stems_sum_to_mix"] = bool(row["mix_mismatch_db"] < tol_db)
            else:
                row["mix_mismatch_db"] = np.nan
                row["stems_sum_to_mix"] = None
            row["ok"] = bool(row["same_shape"] and row["gains_match"] and row["stems_sum_to_mix"] is not False)
        except Exception as e:  # keep going; the error goes in the table
            row["error"] = f"{type(e).__name__}: {e}"
        return row


# helpers
def track_origin(track: str) -> str:
    if any(name in track for name in REAL_DATASETS):
        return "real"
    if any(name in track for name in SYNTHETIC_DATASETS):
        return "synthetic"
    return "unknown"


def mean_hearing_loss(listener: Listener) -> float:
    return float(np.mean([listener.audiogram_left.levels, listener.audiogram_right.levels]))


def clip_stats(signal: np.ndarray, sr: int) -> dict:
    import pyloudnorm as pyln
    from scipy.signal import stft

    mono = signal.mean(axis=1)
    rms = np.sqrt(np.mean(mono**2)) + 1e-12
    peak = np.max(np.abs(signal)) + 1e-12
    try:
        lufs = pyln.Meter(sr).integrated_loudness(signal)
    except ValueError:  # clip shorter than the 400 ms loudness block
        lufs = np.nan
    f, _, z = stft(mono, fs=sr, nperseg=2048)
    mag = np.abs(z).mean(axis=1)
    centroid = float(np.sum(f * mag) / (np.sum(mag) + 1e-12))
    return {
        "lufs": float(lufs) if np.isfinite(lufs) else np.nan,
        "peak_db": float(20 * np.log10(peak)),
        "crest_db": float(20 * np.log10(peak / rms)),
        "centroid_hz": centroid,
    }


def stem_bleed(stems: dict, sr: int, max_lag_ms: float = 50.0) -> dict:
    from scipy.signal import correlate, resample_poly as rp

    mono = {k: rp(s.mean(axis=1), 1, 4) for k, s in stems.items()}  # 44.1 kHz -> ~11 kHz for speed
    lag = int(max_lag_ms / 1000 * sr / 4)
    out = {}
    for k, x in mono.items():
        others = sum(v for j, v in mono.items() if j != k)
        if isinstance(others, int):  # only one stem
            out[k] = np.nan
            continue
        c = correlate(x, others, mode="full", method="fft")
        mid = len(others) - 1
        c = c[mid - lag : mid + lag + 1]
        out[k] = float(np.max(np.abs(c)) / (np.linalg.norm(x) * np.linalg.norm(others) + 1e-12))
    return out
