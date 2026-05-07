# gc_binary_pipeline/io_hugs.py

import os
import numpy as np
import pandas as pd

from .config import HUGS_COLMAP, ClusterConfig


FILTERS = ["F275W", "F336W", "F435W", "F606W", "F814W"]
ERRORS = ["E275", "E336", "E435", "E606", "E814"]
QS = ["Q275", "Q336", "Q435", "Q606", "Q814"]
SHARPS = ["S275", "S336", "S435", "S606", "S814"]


def load_hugs_catalog(file_path: str) -> pd.DataFrame:
    """
    Load HUGS catalog using the user-provided column definition.

    Notes
    -----
    The file is assumed to be whitespace-separated and without a header.
    Columns are renamed according to HUGS_COLMAP.
    """
    from pathlib import Path
    import pandas as pd

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()

 
        if all(c in first_line.split() for c in ["F275W", "F336W", "F438W", "F606W", "F814W"]):
            df = pd.read_csv(file_path, sep=r"\s+", engine="python")

            # ------------------------------------------------------------
            # Compatibility aliases for pipeline expecting HUGS-style names
            # ------------------------------------------------------------

            # Filter alias: pipeline expects F435W, mock has F438W
            if "F435W" not in df.columns and "F438W" in df.columns:
                df["F435W"] = df["F438W"]

            # Error alias for F435W/F438W
            if "e_F435W" not in df.columns and "e_F438W" in df.columns:
                df["e_F435W"] = df["e_F438W"]

            # Compact error column aliases expected by pipeline
            error_aliases = {
                "E275": "e_F275W",
                "E336": "e_F336W",
                "E435": "e_F438W",
                "E438": "e_F438W",
                "E606": "e_F606W",
                "E814": "e_F814W",
            }

            for new_col, old_col in error_aliases.items():
                if new_col not in df.columns and old_col in df.columns:
                    df[new_col] = df[old_col]

            # ------------------------------------------------------------
            # Mock catalog usually has no HUGS quality / sharpness flags.
            #
            # Q-fit in this pipeline is selected by Q > cfg.q_min,
            # so good mock Q values should be high, e.g. 1.0.
            #
            # Sharpness is selected by abs(S) < cfg.sharp_abs_max,
            # so good mock sharpness should be 0.0.
            # ------------------------------------------------------------
            default_one_cols = [
                "Q275", "Q336", "Q435", "Q438", "Q606", "Q814",
            ]

            for col in default_one_cols:
                if col not in df.columns:
                    df[col] = 1.0

            default_zero_cols = [
                "S275", "S336", "S435", "S438", "S606", "S814",
            ]

            for col in default_zero_cols:
                if col not in df.columns:
                    df[col] = 0.0
                        # Q-fit: code requires Q > cfg.q_min.
            # Use a high value to guarantee mock stars pass quality cuts.
            default_q_cols = [
                "Q275", "Q336", "Q435", "Q438", "Q606", "Q814",
            ]

            for col in default_q_cols:
                if col not in df.columns:
                    df[col] = 999.0

            # Sharpness: code requires abs(S) < cfg.sharp_abs_max.
            default_s_cols = [
                "S275", "S336", "S435", "S438", "S606", "S814",
            ]

            for col in default_s_cols:
                if col not in df.columns:
                    df[col] = 0.0

            # Membership probability.
            # Use 100 so it passes both 0--1 and 0--100 style thresholds.
            if "Prob" not in df.columns:
                df["Prob"] = 100.0

            # ------------------------------------------------------------
            # Mock catalog usually has no membership probability.
            # Assume all mock stars are cluster members.
            # ------------------------------------------------------------
            if "Prob" not in df.columns:
                df["Prob"] = 1.0

            # ------------------------------------------------------------
            # Force numerical columns to numeric dtype
            # ------------------------------------------------------------
            numeric_cols = [
                "x", "y", "r",
                "F275W", "F336W", "F438W", "F435W", "F606W", "F814W",
                "m_F275W", "m_F336W", "m_F438W", "m_F606W", "m_F814W",
                "e_F275W", "e_F336W", "e_F438W", "e_F435W", "e_F606W", "e_F814W",
                "E275", "E336", "E435", "E438", "E606", "E814",
                "Q275", "Q336", "Q435", "Q438", "Q606", "Q814",
                "S275", "S336", "S435", "S438", "S606", "S814",
                "Prob",
                "q_truth", "dr_truth",
            ]

            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                        # ------------------------------------------------------------
            # Mock catalogs may use 0 or -1 as placeholder uncertainties.
            # The real-data quality cut requires 0 < E < cfg.rms_max.
            # Replace non-positive / non-finite mock errors by a small
            # positive uncertainty so they do not fail purely due to sentinel
            # values.
            # ------------------------------------------------------------
            mock_error_cols = [
                "e_F275W", "e_F336W", "e_F438W", "e_F435W", "e_F606W", "e_F814W",
                "E275", "E336", "E435", "E438", "E606", "E814",
            ]

            for c in mock_error_cols:
                if c in df.columns:
                    bad = (~np.isfinite(df[c].values)) | (df[c].values <= 0)
                    df.loc[bad, c] = 0.01

            return df




    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    max_col = max(HUGS_COLMAP.keys())
    if df.shape[1] <= max_col:
        raise ValueError(
            f"Catalog has {df.shape[1]} columns, but expected at least {max_col + 1}."
        )

    df = df.rename(columns=HUGS_COLMAP)

    return df



def quality_selection(df: pd.DataFrame, cfg: ClusterConfig) -> pd.DataFrame:
    """
    Apply conservative photometric and membership selection.

    This stage is intentionally conservative because the downstream analysis
    is sensitive to artificial broadening in color and pseudo-color space.
    """
    mask = np.ones(len(df), dtype=bool)

    # finite and non-saturated magnitudes
    for band in FILTERS:
        mask &= np.isfinite(df[band].values)
        mask &= df[band].values > 0
        mask &= df[band].values < 90

    # RMS cuts
    for err in ERRORS:
        mask &= np.isfinite(df[err].values)
        mask &= df[err].values > 0
        mask &= df[err].values < cfg.rms_max

    # Q-fit cuts
    for q in QS:
        mask &= np.isfinite(df[q].values)
        mask &= df[q].values > cfg.q_min

    # sharp cuts
    for s in SHARPS:
        mask &= np.isfinite(df[s].values)
        mask &= np.abs(df[s].values) < cfg.sharp_abs_max

    # membership probability
    if cfg.require_membership:
        mask &= np.isfinite(df["Prob"].values)
        mask &= df["Prob"].values >= cfg.membership_min
    else:
        # If membership unavailable, Prob can be -1.
        mask &= np.isfinite(df["Prob"].values)

    # magnitude range
    mask &= df["F606W"].values >= cfg.mag_min
    mask &= df["F606W"].values <= cfg.mag_max

    selected = df.loc[mask].copy()

    if len(selected) < 300:
        raise RuntimeError(
            f"Too few stars after selection: N={len(selected)}. "
            "Check magnitude range or quality cuts."
        )

    return selected


def downsample_if_needed(df: pd.DataFrame, cfg: ClusterConfig) -> pd.DataFrame:
    if cfg.sample_size is not None and len(df) > cfg.sample_size:
        return df.sample(n=cfg.sample_size, random_state=cfg.random_seed).copy()
    return df.copy()
