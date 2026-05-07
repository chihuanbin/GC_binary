# gc_binary_pipeline/differential_reddening.py

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import NearestNeighbors

from .config import EXTINCTION_COEFF, ClusterConfig


def _ridge_line(mag, color, bins=30, min_count=30, smoothing=0.0005):
    mag = np.asarray(mag)
    color = np.asarray(color)

    edges = np.linspace(np.nanmin(mag), np.nanmax(mag), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    vals = []
    cens = []

    for lo, hi, c in zip(edges[:-1], edges[1:], centers):
        m = (mag >= lo) & (mag < hi) & np.isfinite(color)
        if np.sum(m) < min_count:
            continue
        vals.append(np.nanmedian(color[m]))
        cens.append(c)

    vals = np.asarray(vals)
    cens = np.asarray(cens)

    if len(cens) < 5:
        raise RuntimeError("Not enough points to construct ridge line.")

    return UnivariateSpline(cens, vals, k=3, s=smoothing)


def estimate_local_delta_ebv(
    data: pd.DataFrame,
    cfg: ClusterConfig,
    color_col: str = "Color_Opt_raw",
    mag_col: str = "F606W",
) -> pd.Series:
    """
    Estimate local differential reddening proxy using optical color residuals.

    We use F606W-F814W residuals relative to a main-sequence ridge line.
    The residual is converted to delta E(B-V) using:
        E(F606W-F814W) = (R606 - R814) * E(B-V)

    This is an empirical local correction and should be described as such.
    """
    if "X" not in data.columns or "Y" not in data.columns:
        raise ValueError("X/Y positions are required for local DR correction.")

    d = data.copy()

    if color_col not in d.columns:
        d[color_col] = d["F606W"] - d["F814W"]

    ridge = _ridge_line(
        d[mag_col].values,
        d[color_col].values,
        bins=cfg.fiducial_bins,
        min_count=cfg.min_bin_count,
        smoothing=cfg.fiducial_smoothing,
    )

    residual = d[color_col].values - ridge(d[mag_col].values)

    coords = d[["X", "Y"]].values
    n_neighbors = min(cfg.dr_n_neighbors, len(d) - 1)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(coords)
    _, ind = nbrs.kneighbors(coords)

    local_color_excess = np.array([
        np.nanmedian(residual[ii]) for ii in ind
    ])

    r606 = EXTINCTION_COEFF["F606W"]
    r814 = EXTINCTION_COEFF["F814W"]
    denom = r606 - r814

    if denom <= 0:
        raise ValueError("Invalid extinction coefficients for F606W/F814W.")

    delta_ebv = local_color_excess / denom

    return pd.Series(delta_ebv, index=d.index, name="delta_EBV")


def apply_differential_reddening_correction(
    data: pd.DataFrame,
    cfg: ClusterConfig,
) -> pd.DataFrame:
    """
    Apply local differential reddening correction to all HUGS bands.

    Corrected magnitudes overwrite the original band columns only after
    preserving raw magnitudes with *_raw names.
    """
    d = data.copy()

    # Preserve raw photometry
    for band in EXTINCTION_COEFF:
        raw_name = f"{band}_raw"
        if raw_name not in d.columns:
            d[raw_name] = d[band].values

    d["Color_Opt_raw"] = d["F606W_raw"] - d["F814W_raw"]

    delta_ebv = estimate_local_delta_ebv(d, cfg)
    d["delta_EBV"] = delta_ebv

    for band, coeff in EXTINCTION_COEFF.items():
        d[band] = d[f"{band}_raw"] - coeff * d["delta_EBV"].values

    return d
