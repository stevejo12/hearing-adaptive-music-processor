"""Cadenza baseline processing: decode -> separate -> gains -> remix -> hearing aid.

Re-uses the Clarity CAD2 Task 2 baseline functions (installed into the image at a
pinned commit) so the output matches the Colab baseline for the same inputs.
Pretrained ConvTasNet weights are downloaded once into the /data volume (models.py).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

from . import models, profile

logger = logging.getLogger(__name__)

INPUT_SR = 44100
REMIX_SR = 32000
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "120"))
DEFAULT_RESIDUAL = 0.1   # baseline add_residual (share of the accompaniment kept in each stem)

# Baseline hearing-aid settings (recipes/cad2/task2/baseline/config.yaml)
CROSSOVER = [353.55, 707.11, 1414.21, 2828.43, 5656.85]
ATTACK = [11, 11, 14, 13, 11, 11]
RELEASE = [80, 80, 80, 80, 100, 100]
THRESHOLD = [-30, -30, -30, -30, -30, -30]

INSTRUMENTS = list(profile.INSTRUMENT_CENTRE_HZ)

_cache: "OrderedDict[str, dict]" = OrderedDict()   # model id -> {instrument: network}
_cache_lock = threading.Lock()
MAX_CACHED_MODELS = int(os.environ.get("MAX_CACHED_MODELS", "2"))


def _device():
    import torch  # imported lazily so the web layer can be tested without torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_loadable(path: str) -> None:
    """Prove an uploaded instrument folder really loads (used when a model is uploaded)."""
    from recipes.cad2.task2.ConvTasNet.local.tasnet import ConvTasNetStereo

    try:
        ConvTasNetStereo.from_pretrained(path)
    except Exception as exc:
        raise ValueError(f"{Path(path).name} could not be loaded as a Cadenza ConvTasNet "
                         f"({str(exc)[:150]})") from exc


def load_models(model_id: str = models.BASELINE_ID) -> dict:
    """Per-instrument separators for one model, loaded once and kept (small LRU cache)."""
    with _cache_lock:
        if model_id in _cache:
            _cache.move_to_end(model_id)
            return _cache[model_id]
        from recipes.cad2.task2.ConvTasNet.local.tasnet import ConvTasNetStereo

        device = _device()
        loaded = {}
        for name, path in models.model_paths(model_id).items():   # all on disk already
            logger.info("Loading %s from %s", name, path)
            loaded[name] = ConvTasNetStereo.from_pretrained(path).to(device).eval()
        _cache[model_id] = loaded
        while len(_cache) > MAX_CACHED_MODELS:
            _cache.popitem(last=False)
        return loaded


def forget_model(model_id: str) -> None:
    with _cache_lock:
        _cache.pop(model_id, None)


def decode_audio(path: Path, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Any audio file -> float32 stereo array (samples, 2) at 44.1 kHz, via ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-t", str(max_seconds),
               "-ac", "2", "-ar", str(INPUT_SR), "-f", "wav", str(wav)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not wav.exists():
            raise ValueError("Could not decode the audio file: " + result.stderr.strip()[-200:])
        audio, sr = sf.read(wav, dtype="float32", always_2d=True)
    if sr != INPUT_SR or audio.shape[0] < INPUT_SR:  # need at least 1 s
        raise ValueError("Audio is too short or has an unexpected format")
    return audio


def flac_to_mp3(src: Path, dst: Path, bitrate: str = "256k") -> None:
    """Convert the processed FLAC to MP3 so every browser can play it."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-codec:a", "libmp3lame",
           "-b:a", bitrate, str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dst.exists():
        raise RuntimeError("MP3 conversion failed: " + result.stderr.strip()[-200:])


def process(before_path: Path, after_path: Path, instruments: list[str],
            gains_db: list[float], levels_l: np.ndarray, levels_r: np.ndarray,
            model_id: str = models.BASELINE_ID, residual: float = DEFAULT_RESIDUAL) -> dict:
    """Run the full baseline on one song. Writes an MP3 (converted from the FLAC) to `after_path`."""
    from clarity.enhancer.multiband_compressor import MultibandCompressor
    from clarity.utils.flac_encoder import save_flac_signal
    from recipes.cad2.task2.baseline.enhance import (
        check_repeated_source, decompose_signal, process_remix_for_listener)
    from recipes.cad2.task2.baseline.evaluate import adjust_level, apply_gains, remix_stems

    separators = load_models(model_id)
    device = _device()
    audio = decode_audio(before_path)

    source_list = {f"source_{i}": name for i, name in enumerate(instruments, 1)}
    gains = {f"source_{i}": float(g) for i, g in enumerate(gains_db, 1)}

    stems = decompose_signal(
        model=separators, signal=audio.T, signal_sample_rate=INPUT_SR, device=device,
        sources_list=source_list, listener=None, add_residual=residual)

    gain_scene = check_repeated_source(gains, source_list)
    stems = apply_gains(stems, INPUT_SR, gain_scene)
    remix = remix_stems(stems)
    remix = adjust_level(remix, gains)

    enhancer = MultibandCompressor(crossover_frequencies=CROSSOVER, sample_rate=INPUT_SR)
    hearing_aid = profile.compressor_params(levels_l, levels_r)
    for ear in hearing_aid.values():
        ear.update(attack=ATTACK, release=RELEASE, threshold=THRESHOLD)
    out = process_remix_for_listener(remix, enhancer, hearing_aid, listener=None)

    after_path.parent.mkdir(parents=True, exist_ok=True)
    flac_path = after_path.with_suffix(".flac")    # Cadenza output format; converted for the browser
    save_flac_signal(signal=out, filename=flac_path, signal_sample_rate=INPUT_SR,
                     output_sample_rate=REMIX_SR, do_clip_signal=True, do_soft_clip=False)
    try:
        flac_to_mp3(flac_path, after_path)
    finally:
        flac_path.unlink(missing_ok=True)
    return {"seconds": round(audio.shape[0] / INPUT_SR, 1)}
