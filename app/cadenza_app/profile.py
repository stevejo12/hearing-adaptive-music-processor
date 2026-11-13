"""Hearing profile helpers: audiogram parsing, hearing-aid settings, default gains.

Pure numpy, no torch or Clarity imports, so it is cheap to test.

ASSUMPTIONS (not part of the Cadenza challenge, to be validated in Colab):
  * `compressor_params()` follows the example in Clarity's MultibandCompressor
    docstring: ratio 4.0 and make-up gain max((HL - 20) / 3, 0) per band. The CAD2
    dataset ships ready-made per-listener values (compressor_params.*.json) that we
    cannot derive from a new audiogram, so this is a stand-in.
  * `default_gains()` is the "simple rule from audiogram" decided for the project:
    instruments that sit in the user's worse-hearing region get a modest boost.
"""

from __future__ import annotations

import numpy as np

# Clarity's default audiogram frequencies (Hz), one level (dB HL) each.
AUDIOGRAM_FREQS = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000], dtype=float)

# Centre frequency of each of the 6 compressor bands (crossovers are
# [250, 500, 1000, 2000, 4000] * sqrt(2) in the baseline config).
BAND_CENTRES = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)

# Rough spectral centre of each supported woodwind (Hz). Tunable.
INSTRUMENT_CENTRE_HZ = {
    "Bassoon": 250.0,
    "Sax": 500.0,
    "Clarinet": 700.0,
    "Oboe": 1000.0,
    "Flute": 1500.0,
}

MIN_HL, MAX_HL = -10.0, 120.0
COMPRESSION_RATIO = 4.0
MAX_AUTO_GAIN_DB = 6.0
AUTO_GAIN_SLOPE = 0.2  # dB of gain per dB of hearing-loss difference


def parse_levels(values) -> np.ndarray:
    """Validate one ear's audiogram: 8 numbers in dB HL (250 Hz ... 8 kHz)."""
    try:
        levels = np.array([float(v) for v in values], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Audiogram values must be numbers") from exc
    if levels.shape != (len(AUDIOGRAM_FREQS),):
        raise ValueError(f"Audiogram needs {len(AUDIOGRAM_FREQS)} values (250 Hz to 8 kHz)")
    if not np.all(np.isfinite(levels)) or levels.min() < MIN_HL or levels.max() > MAX_HL:
        raise ValueError(f"Audiogram values must be between {MIN_HL:.0f} and {MAX_HL:.0f} dB HL")
    return levels


def hl_at(levels: np.ndarray, freqs) -> np.ndarray:
    """Hearing loss at arbitrary frequencies (linear interpolation on a log axis)."""
    return np.interp(np.log(np.asarray(freqs, dtype=float)), np.log(AUDIOGRAM_FREQS), levels)


def compressor_params(levels_l: np.ndarray, levels_r: np.ndarray) -> dict[str, dict]:
    """Per-ear compressor ratio and make-up gain (6 bands) for the hearing aid stage."""
    params = {}
    for ear, levels in (("left", levels_l), ("right", levels_r)):
        hl = hl_at(levels, BAND_CENTRES)
        params[ear] = {
            "ratio": COMPRESSION_RATIO,
            "makeup_gain": [float(g) for g in np.maximum((hl - 20.0) / 3.0, 0.0)],
        }
    return params


def default_gains(instruments: list[str], levels_l: np.ndarray, levels_r: np.ndarray) -> list[float]:
    """Default per-instrument gains (dB), same order as `instruments`.

    Boost instruments whose spectral centre falls where the listener's average
    hearing loss is worse than across the instruments in this piece, cut the others.
    Gains are zero-mean and limited to +/- MAX_AUTO_GAIN_DB.
    """
    unknown = [i for i in instruments if i not in INSTRUMENT_CENTRE_HZ]
    if unknown:
        raise ValueError(f"Unsupported instrument(s): {', '.join(unknown)}")
    mean_levels = (np.asarray(levels_l) + np.asarray(levels_r)) / 2.0
    scores = hl_at(mean_levels, [INSTRUMENT_CENTRE_HZ[i] for i in instruments])
    gains = np.clip(AUTO_GAIN_SLOPE * (scores - scores.mean()), -MAX_AUTO_GAIN_DB, MAX_AUTO_GAIN_DB)
    return [round(float(g), 1) for g in gains]
