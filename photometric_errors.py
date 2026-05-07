# gc_binary_pipeline/photometric_errors.py

import numpy as np
import pandas as pd

from .config import ClusterConfig


def compute_chromosome_errors(
    data: pd.DataFrame,
    fiducials: dict,
    cfg: ClusterConfig,
) -> pd.DataFrame:
    """
    Propagate HUGS photometric RMS values into chromosome-map coordinates.

    C_pseudo = F275W - 2 F336W + F435W
    sigma(C_pseudo)^2 = sigma275^2 + 4 sigma336^2 + sigma435^2

    C_opt = F606W - F814W
    sigma(C_opt)^2 = sigma606^2 + sigma814^2

    The normalized errors are approximated by dividing by the local
    two-fiducial separation.
    """
    d = data.copy()
    mag = d["F606W"].values

    sig_pseudo = np.sqrt(
        d["E275"].values**2
        + 4.0 * d["E336"].values**2
        + d["E435"].values**2
    )

    sig_opt = np.sqrt(
        d["E606"].values**2
        + d["E814"].values**2
    )

    sep_pseudo = np.abs(
        fiducials["pseudo_red"](mag) - fiducials["pseudo_blue"](mag)
    )
    sep_opt = np.abs(
        fiducials["opt_red"](mag) - fiducials["opt_blue"](mag)
    )

    floor = 1e-5

    d["e_Delta_Pseudo"] = sig_pseudo / np.maximum(sep_pseudo, floor)
    d["e_Delta_Opt"] = sig_opt / np.maximum(sep_opt, floor)

    # Add a systematic floor to prevent unrealistically small errors
    d["e_Delta_Pseudo"] = np.sqrt(
        d["e_Delta_Pseudo"].values**2 + cfg.delta_error_floor**2
    )

    d["e_Delta_Opt"] = np.sqrt(
        d["e_Delta_Opt"].values**2 + cfg.delta_error_floor**2
    )

    finite = (
        np.isfinite(d["e_Delta_Pseudo"].values)
        & np.isfinite(d["e_Delta_Opt"].values)
        & (d["e_Delta_Pseudo"].values > 0)
        & (d["e_Delta_Opt"].values > 0)
    )

    return d.loc[finite].copy()
