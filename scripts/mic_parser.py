"""
MIC value parser for ATLAS / SENTRY / GEARS surveillance data.

ATLAS uses abbreviated dilution labels (0.03 = 2^-5, 0.06 = 2^-4, 0.12 = 2^-3 ...).
We map each label to its exact log2 integer index, so dilution-step arithmetic is
clean integer math.

Returns per cell:
  mic_obs      : raw string as in source
  mic_label    : numeric label (0.06, 0.5, 16 ...) the lab reported
  log2_idx     : integer log2 index of that label (e.g. 0.06 -> -4)
  log2_lower   : interval LOWER bound in log2 space (-inf if left-cens at lowest)
  log2_upper   : interval UPPER bound in log2 space (+inf if right-cens at highest)
  cens_type    : 'exact' (interval (i-1, i]), 'left' ((-inf, i]), 'right' ((i, +inf))

An exact reading "label=L" means growth at L/2 but inhibition at L,
so true MIC ∈ (L/2, L]  ->  log2 interval (idx-1, idx].
"""
from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd

LABEL_TO_LOG2 = {
    "0.001": -10,  # below-LOD style; ATLAS uses this for a few drugs
    "0.002": -9,
    "0.004": -8,
    "0.008": -7,
    "0.015": -6,
    "0.03": -5,
    "0.06": -4,
    "0.12": -3,
    "0.25": -2,
    "0.5": -1,
    "1": 0, "1.0": 0,
    "2": 1, "2.0": 1,
    "4": 2, "4.0": 2,
    "8": 3, "8.0": 3,
    "16": 4, "16.0": 4,
    "32": 5, "32.0": 5,
    "64": 6, "64.0": 6,
    "128": 7, "128.0": 7,
    "256": 8, "256.0": 8,
    "512": 9, "512.0": 9,
}

_RE = re.compile(r"^\s*([<>]=?)?\s*([0-9]+(?:\.[0-9]+)?)\s*$")

_BLANK = {
    "mic_obs": np.nan, "mic_label": np.nan, "log2_idx": np.nan,
    "log2_lower": np.nan, "log2_upper": np.nan, "cens_type": np.nan,
}


def _label_to_idx(num_str: str) -> int | None:
    if num_str in LABEL_TO_LOG2:
        return LABEL_TO_LOG2[num_str]
    try:
        v = float(num_str)
    except ValueError:
        return None
    if v <= 0:
        return None
    return int(round(math.log2(v)))


def parse_one(raw) -> dict:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return dict(_BLANK)
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return dict(_BLANK)
    m = _RE.match(s)
    if not m:
        return dict(_BLANK)
    op = m.group(1) or ""
    num_raw = m.group(2)
    # normalize: '16.0' -> '16', '0.060' -> '0.06'
    if "." in num_raw:
        num_norm = num_raw.rstrip("0").rstrip(".")
        if num_norm == "" or num_norm.startswith("."):
            num_norm = "0" + num_norm if num_norm else "0"
    else:
        num_norm = num_raw
    idx = _label_to_idx(num_norm)
    if idx is None:
        idx = _label_to_idx(num_raw)
    if idx is None:
        return dict(_BLANK)
    out = {"mic_obs": s, "mic_label": float(num_raw), "log2_idx": idx}
    if op == "":
        out["log2_lower"] = float(idx) - 1.0
        out["log2_upper"] = float(idx)
        out["cens_type"] = "exact"
    elif op in ("<", "<="):
        out["log2_lower"] = -np.inf
        out["log2_upper"] = float(idx)
        out["cens_type"] = "left"
    elif op in (">", ">="):
        out["log2_lower"] = float(idx)
        out["log2_upper"] = np.inf
        out["cens_type"] = "right"
    else:
        return dict(_BLANK)
    return out


def parse_series(s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([parse_one(v) for v in s], index=s.index)


if __name__ == "__main__":
    tests = ["<=0.06", "0.5", ">16", "2", " 0.03 ", ">=64", "nan", None, "", "abc", "0.12", "128", "0.001"]
    for t in tests:
        r = parse_one(t)
        print(f"  {repr(t):>14}  ->  idx={r['log2_idx']}  lower={r['log2_lower']}  upper={r['log2_upper']}  cens={r['cens_type']}")
