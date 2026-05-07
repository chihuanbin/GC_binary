# gc_binary_pipeline/chromosome_map.py

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.mixture import GaussianMixture

from .config import ClusterConfig


def add_colors(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add pseudo-color and optical color.

    HUGS catalog here provides F435W, not F438W.
    We therefore define:
        C_pseudo = F275W - 2 F336W + F435W
        C_opt    = F606W - F814W

    This should be explicitly stated in the manuscript.
    """
    d = data.copy()
    d["C_pseudo"] = d["F275W"] - 2.0 * d["F336W"] + d["F435W"]
    d["Color_Opt"] = d["F606W"] - d["F814W"]
    return d


def build_two_fiducials(
    mag,
    color,
    cfg: ClusterConfig,
):
    """
    Build blue and red fiducial boundaries using percentiles in magnitude bins.
    """
    mag = np.asarray(mag)
    color = np.asarray(color)

    edges = np.linspace(np.nanmin(mag), np.nanmax(mag), cfg.fiducial_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    blue_vals = []
    red_vals = []
    valid_centers = []

    for lo, hi, c in zip(edges[:-1], edges[1:], centers):
        m = (
            (mag >= lo)
            & (mag < hi)
            & np.isfinite(color)
        )
        if np.sum(m) < cfg.min_bin_count:
            continue

        vals = color[m]
        blue_vals.append(np.nanpercentile(vals, cfg.blue_percentile))
        red_vals.append(np.nanpercentile(vals, cfg.red_percentile))
        valid_centers.append(c)

    blue_vals = np.asarray(blue_vals)
    red_vals = np.asarray(red_vals)
    valid_centers = np.asarray(valid_centers)

    if len(valid_centers) < 5:
        raise RuntimeError(
            "Not enough valid magnitude bins to build two fiducials."
        )

    blue_spline = UnivariateSpline(
        valid_centers,
        blue_vals,
        k=3,
        s=cfg.fiducial_smoothing,
    )

    red_spline = UnivariateSpline(
        valid_centers,
        red_vals,
        k=3,
        s=cfg.fiducial_smoothing,
    )

    return blue_spline, red_spline


def normalize_two_fiducial(
    mag,
    color,
    blue_spline,
    red_spline,
    center: bool = True,
    allow_extrapolation: bool = False,
):
    """
    Normalize a color using blue and red fiducials.

    Important:
    By default, this function does NOT extrapolate outside the magnitude
    range covered by the fiducial knots. Extrapolation can produce extremely
    large and unphysical Delta values, especially for synthetic binaries that
    become brighter than the single-star magnitude range.
    """
    mag = np.asarray(mag, dtype=float)
    color = np.asarray(color, dtype=float)

    delta = np.full_like(mag, np.nan, dtype=float)

    finite = np.isfinite(mag) & np.isfinite(color)

    if not allow_extrapolation:
        blue_knots = blue_spline.get_knots()
        red_knots = red_spline.get_knots()

        mag_lo = max(np.nanmin(blue_knots), np.nanmin(red_knots))
        mag_hi = min(np.nanmax(blue_knots), np.nanmax(red_knots))

        finite &= (mag >= mag_lo) & (mag <= mag_hi)

    if np.sum(finite) == 0:
        return delta

    blue = blue_spline(mag[finite])
    red = red_spline(mag[finite])

    denom = red - blue

    good = (
        np.isfinite(blue)
        & np.isfinite(red)
        & np.isfinite(denom)
        & (np.abs(denom) > 1e-4)
    )

    tmp = np.full(np.sum(finite), np.nan, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        tmp[good] = (color[finite][good] - blue[good]) / denom[good]

    if center:
        tmp[good] = tmp[good] - 0.5

    delta[finite] = tmp

    bad = ~np.isfinite(delta)
    delta[bad] = np.nan

    return delta



def construct_chromosome_map(
    data: pd.DataFrame,
    cfg: ClusterConfig,
):
    """
    Construct chromosome-map-like coordinates using two-fiducial normalization.

    Returns
    -------
    data_out : DataFrame
        With Delta_Pseudo and Delta_Opt.
    fiducials : dict
        Spline fiducials needed to transform synthetic binaries.
    """
    d = add_colors(data)

    mag = d["F606W"].values

    p_blue, p_red = build_two_fiducials(
        mag,
        d["C_pseudo"].values,
        cfg,
    )

    o_blue, o_red = build_two_fiducials(
        mag,
        d["Color_Opt"].values,
        cfg,
    )

    d["Delta_Pseudo"] = normalize_two_fiducial(
        mag,
        d["C_pseudo"].values,
        p_blue,
        p_red,
        center=True,
    )

    d["Delta_Opt"] = normalize_two_fiducial(
        mag,
        d["Color_Opt"].values,
        o_blue,
        o_red,
        center=True,
    )

    finite = (
    np.isfinite(d["Delta_Pseudo"].values)
    & np.isfinite(d["Delta_Opt"].values)
    & (np.abs(d["Delta_Pseudo"].values) < cfg.delta_abs_max)
    & (np.abs(d["Delta_Opt"].values) < cfg.delta_abs_max)
    )

    n_removed = len(d) - int(np.sum(finite))
    if n_removed > 0:
        print(
            f"Removed {n_removed} stars with extreme chromosome-map coordinates "
            f"(|Delta| >= {cfg.delta_abs_max})."
        )

    d = d.loc[finite].copy()


    fiducials = {
        "pseudo_blue": p_blue,
        "pseudo_red": p_red,
        "opt_blue": o_blue,
        "opt_red": o_red,
    }

    return d, fiducials


def initial_population_labels(
    data: pd.DataFrame,
    cfg: ClusterConfig,
) -> pd.DataFrame:
    """
    Conservative initial 1G/2G labels using a two-component GMM.

    These labels are used only to construct empirical templates.
    Final inference is performed probabilistically.
    """
    d = data.copy()

    X = d[["Delta_Pseudo", "Delta_Opt"]].values

    gmm = GaussianMixture(
        n_components=cfg.init_gmm_components,
        covariance_type="full",
        random_state=cfg.random_seed,
        reg_covar=1e-4,
    )
    gmm.fit(X)

    prob = gmm.predict_proba(X)

    # The assignment of 1G/2G depends on convention.
    # Here we use Delta_Pseudo ordering.
    means = gmm.means_[:, 0]
    comp_low = int(np.argmin(means))
    comp_high = int(np.argmax(means))

    # Adopt:
    # low Delta_Pseudo -> 2G
    # high Delta_Pseudo -> 1G
    # This must be checked cluster-by-cluster.
    comp_2g = comp_low
    comp_1g = comp_high

    d["p_init_1g"] = prob[:, comp_1g]
    d["p_init_2g"] = prob[:, comp_2g]

    d["pop_init"] = 0
    d.loc[d["p_init_1g"] > cfg.init_prob_threshold, "pop_init"] = 1
    d.loc[d["p_init_2g"] > cfg.init_prob_threshold, "pop_init"] = 2

    return d
