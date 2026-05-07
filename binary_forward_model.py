# gc_binary_pipeline/binary_forward_model.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

from .config import ClusterConfig
from .chromosome_map import normalize_two_fiducial


def _pseudo_red_band(df: pd.DataFrame) -> str:
    """
    Return the blue/near-UV band used in the chromosome-map pseudo-color.

    Some code/data use F435W, while HUGS-like catalogs often use F438W.
    The pseudo-color is:
        C = F275W - 2 F336W + F435W/F438W
    """
    if "F435W" in df.columns:
        return "F435W"
    if "F438W" in df.columns:
        return "F438W"

    raise KeyError(
        "Neither F435W nor F438W found in DataFrame. "
        "Cannot construct chromosome-map pseudo-color."
    )


def _photometric_bands(df: pd.DataFrame):
    """
    Return the photometric bands needed for binary flux addition.
    Supports either F435W or F438W.
    """
    pseudo_band = _pseudo_red_band(df)

    bands = [
        "F275W",
        "F336W",
        pseudo_band,
        "F606W",
        "F814W",
    ]

    missing = [b for b in bands if b not in df.columns]
    if missing:
        raise KeyError(f"Missing required photometric bands: {missing}")

    return bands


def combine_magnitudes(m1, m2):
    """
    Combine two magnitudes by adding fluxes.
    """
    return -2.5 * np.log10(
        10.0 ** (-0.4 * m1) + 10.0 ** (-0.4 * m2)
    )


def secondary_f606_from_q(
    primary_f606,
    q,
    alpha=4.0,
):
    """
    Approximate secondary F606W magnitude using L ~ M^alpha.

    m2 = m1 - 2.5 log10(q^alpha)

    Since q < 1, log10(q) < 0, so m2 is fainter than m1.
    """
    return primary_f606 - 2.5 * alpha * np.log10(q)


def _fit_sequence_neighbor(clean_pop: pd.DataFrame):
    """
    Build nearest-neighbor search in F606W for empirical sequence interpolation.
    """
    mags = clean_pop["F606W"].values.reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(mags)
    return nn


def _draw_secondary_from_population(
    clean_pop: pd.DataFrame,
    nn: NearestNeighbors,
    target_f606,
):
    """
    Select empirical star from a population sequence closest to target F606W.
    """
    _, ind = nn.kneighbors(np.asarray(target_f606).reshape(-1, 1))
    ind = ind[:, 0]
    return clean_pop.iloc[ind].reset_index(drop=True)


def transform_mags_to_chromosome(mags: pd.DataFrame, fiducials: dict):
    """
    Transform synthetic magnitudes through the same chromosome-map normalization.
    """
    pseudo_band = _pseudo_red_band(mags)

    c_pseudo = mags["F275W"] - 2.0 * mags["F336W"] + mags[pseudo_band]
    c_opt = mags["F606W"] - mags["F814W"]
    mag = mags["F606W"].values

    d_pseudo = normalize_two_fiducial(
        mag,
        c_pseudo.values,
        fiducials["pseudo_blue"],
        fiducials["pseudo_red"],
        center=True,
    )

    d_opt = normalize_two_fiducial(
        mag,
        c_opt.values,
        fiducials["opt_blue"],
        fiducials["opt_red"],
        center=True,
    )

    pts = np.vstack([d_pseudo, d_opt]).T
    finite = np.isfinite(pts).all(axis=1)

    return pts[finite]


def _get_mock_truth_population_masks(data: pd.DataFrame):
    """
    Return boolean masks for truth 1G/2G populations in mock catalogs.

    Expected mock labels:
        pop_truth == 1  -> 1G
        pop_truth == 2  -> 2G
        pop_truth == False / 0 / NaN -> field or non-cluster

    This function is conservative:
    only numeric 1 and 2 are treated as valid cluster populations.
    """
    if "pop_truth" not in data.columns:
        return None, None

    pop = pd.to_numeric(data["pop_truth"], errors="coerce")

    mask_1g = pop == 1
    mask_2g = pop == 2

    if mask_1g.sum() == 0 or mask_2g.sum() == 0:
        return None, None

    return mask_1g, mask_2g


def _select_truth_clean_populations(
    data: pd.DataFrame,
    cfg: ClusterConfig,
):
    """
    Select clean 1G/2G template stars using mock truth labels.

    This is intended for mock recovery tests. It avoids using the
    chromosome-map automatic 1G/2G split, which can be badly mismatched
    to simulated catalogs.

    By default, if binary_type_truth exists, only truth single stars
    are used for empirical single-star templates.
    """
    mask_1g, mask_2g = _get_mock_truth_population_masks(data)

    if mask_1g is None or mask_2g is None:
        return None, None

    base = (
        np.isfinite(data["Delta_Pseudo"])
        & np.isfinite(data["Delta_Opt"])
        & (np.abs(data["Delta_Pseudo"]) < cfg.template_delta_abs_max)
        & (np.abs(data["Delta_Opt"]) < cfg.template_delta_abs_max)
        & np.isfinite(data["F606W"])
        & (data["F606W"] >= cfg.mag_min)
        & (data["F606W"] <= cfg.mag_max)
    )

    # Require finite photometry in the bands used by the forward model.
    bands = _photometric_bands(data)
    for band in bands:
        base = base & np.isfinite(data[band])

    use_truth_singles = getattr(
        cfg,
        "use_truth_singles_for_templates",
        True,
    )

    if use_truth_singles and "binary_type_truth" in data.columns:
        binary_type = pd.to_numeric(
            data["binary_type_truth"],
            errors="coerce",
        )

        # binary_type_truth == 0 means truth single star.
        single_mask = binary_type.fillna(-999) == 0

        mask_1g = mask_1g & single_mask
        mask_2g = mask_2g & single_mask

    clean_1g = data.loc[base & mask_1g].copy()
    clean_2g = data.loc[base & mask_2g].copy()

    if len(clean_1g) == 0 or len(clean_2g) == 0:
        return None, None

    return clean_1g, clean_2g


def generate_binary_template(
    primary_pop: pd.DataFrame,
    secondary_pop: pd.DataFrame,
    fiducials: dict,
    cfg: ClusterConfig,
    n_template: int,
):
    """
    Generate unresolved binary template by flux addition.

    This is an empirical forward model:
    - primaries are drawn from a clean empirical population sequence;
    - q is drawn uniformly over [q_min_binary, q_max_binary];
    - secondary F606W magnitude is estimated from q using L~M^alpha;
    - secondary multi-band magnitudes are taken from the empirical sequence
      closest to the target F606W magnitude;
    - fluxes are added in each filter;
    - the resulting system is passed through the same chromosome-map transform.
    """
    rng = np.random.default_rng(cfg.random_seed)

    if len(primary_pop) < 50 or len(secondary_pop) < 50:
        raise RuntimeError(
            "Too few clean stars to build binary template: "
            f"N_primary={len(primary_pop)}, N_secondary={len(secondary_pop)}"
        )

    bands_primary = _photometric_bands(primary_pop)
    bands_secondary = _photometric_bands(secondary_pop)

    # Require the same band set.
    if set(bands_primary) != set(bands_secondary):
        raise RuntimeError(
            f"Primary and secondary populations have different band sets: "
            f"{bands_primary} vs {bands_secondary}"
        )

    bands = bands_primary

    # Avoid primaries too close to the bright limit. An equal-mass binary is
    # ~0.75 mag brighter than the primary, so primaries near cfg.mag_min can
    # produce systems outside the fiducial magnitude range.
    primary_allowed = primary_pop[
        (primary_pop["F606W"] >= cfg.mag_min + cfg.binary_primary_mag_buffer)
        & (primary_pop["F606W"] <= cfg.mag_max)
    ].copy()

    if len(primary_allowed) < 50:
        print(
            "Warning: too few primaries after bright-end buffer. "
            "Falling back to full primary population."
        )
        primary_allowed = primary_pop.copy()

    primary = primary_allowed.sample(
        n=n_template,
        replace=True,
        random_state=cfg.random_seed,
    ).reset_index(drop=True)

    q = rng.uniform(cfg.q_min_binary, cfg.q_max_binary, size=n_template)

    target_f606_secondary = secondary_f606_from_q(
        primary["F606W"].values,
        q,
        alpha=cfg.mass_luminosity_alpha,
    )

    nn_secondary = _fit_sequence_neighbor(secondary_pop)

    secondary = _draw_secondary_from_population(
        secondary_pop,
        nn_secondary,
        target_f606_secondary,
    )

    combined = pd.DataFrame()

    for band in bands:
        combined[band] = combine_magnitudes(
            primary[band].values,
            secondary[band].values,
        )

    # Keep only synthetic unresolved systems that remain inside the analysis
    # magnitude range.
    mag_ok = (
        np.isfinite(combined["F606W"].values)
        & (combined["F606W"].values >= cfg.mag_min)
        & (combined["F606W"].values <= cfg.mag_max)
    )

    combined = combined.loc[mag_ok].reset_index(drop=True)

    pts = transform_mags_to_chromosome(combined, fiducials)

    # Robust clipping in normalized coordinates.
    pts = pts[
        np.isfinite(pts).all(axis=1)
        & (np.abs(pts[:, 0]) < cfg.template_delta_abs_max)
        & (np.abs(pts[:, 1]) < cfg.template_delta_abs_max)
    ]

    min_valid_binary_template = getattr(
        cfg,
        "min_valid_binary_template",
        500,
    )

    if len(pts) < min_valid_binary_template:
        raise RuntimeError(
            f"Too few valid synthetic binaries after clipping: N={len(pts)}, "
            f"min_valid_binary_template={min_valid_binary_template}. "
            "Consider increasing n_binary_template or relaxing "
            "template_delta_abs_max."
        )

    return pts


def fit_template_gmm(
    points,
    n_components,
    cfg: ClusterConfig,
):
    """
    Approximate template point cloud with a Gaussian mixture.

    Mock-friendly version:
    - remove non-finite points;
    - adaptively reduce the number of GMM components when the template
      sample is small;
    - use reg_covar to avoid singular covariance matrices.
    """
    points = np.asarray(points)

    if points.ndim != 2:
        raise RuntimeError(
            f"Template GMM points must be a 2D array, got shape={points.shape}"
        )

    points = points[np.isfinite(points).all(axis=1)]
    N = len(points)

    if N < 10:
        raise RuntimeError(
            f"Too few points to fit GMM even with adaptive K: N={N}"
        )

    requested_K = int(n_components)

    # Minimum support per Gaussian component.
    # For example:
    # N=67 and requested_K=4 with min_points_per_component=25
    # gives final_K=2.
    min_points_per_component = getattr(
        cfg,
        "min_points_per_gmm_component",
        25,
    )

    adaptive_K = max(1, N // min_points_per_component)
    final_K = min(requested_K, adaptive_K)

    # Extra safety: sklearn requires n_components <= N.
    final_K = min(final_K, N)

    if final_K < requested_K:
        print(
            f"Reducing GMM components from K={requested_K} to K={final_K} "
            f"because only N={N} template points are available."
        )

    gmm = GaussianMixture(
        n_components=final_K,
        covariance_type="full",
        random_state=getattr(cfg, "random_seed", 42),
        reg_covar=getattr(cfg, "gmm_reg_covar", 1e-5),
    )

    gmm.fit(points)

    return {
        "weights": gmm.weights_.astype(float),
        "means": gmm.means_.astype(float),
        "covs": gmm.covariances_.astype(float),
    }


def build_field_template(
    df_all_transformed: pd.DataFrame,
    cfg: ClusterConfig,
):
    """
    Build field template from low-membership stars if available.

    Requires low-membership stars to have already been transformed into
    Delta_Pseudo/Delta_Opt.
    """
    if "Prob" not in df_all_transformed.columns:
        return None

    field = df_all_transformed[
        np.isfinite(df_all_transformed["Prob"])
        & (df_all_transformed["Prob"] >= 0)
        & (df_all_transformed["Prob"] < 50)
    ].copy()

    if len(field) < 300:
        return None

    pts = field[["Delta_Pseudo", "Delta_Opt"]].values

    return fit_template_gmm(
        pts,
        n_components=cfg.gmm_components_field,
        cfg=cfg,
    )


def robust_clean_population(
    data: pd.DataFrame,
    pop_id: int,
    cfg: ClusterConfig,
    prob_col: str,
    min_prob: float = 0.90,
):
    """
    Select a clean empirical population sequence for template construction.

    This removes:
    - uncertain initial GMM assignments;
    - extreme chromosome-map outliers;
    - broad tails likely caused by photometric scatter or binaries.
    """
    d = data[
        (data["pop_init"] == pop_id)
        & np.isfinite(data[prob_col])
        & (data[prob_col] >= min_prob)
        & np.isfinite(data["Delta_Pseudo"])
        & np.isfinite(data["Delta_Opt"])
        & (np.abs(data["Delta_Pseudo"]) < cfg.template_delta_abs_max)
        & (np.abs(data["Delta_Opt"]) < cfg.template_delta_abs_max)
    ].copy()

    if len(d) < 50:
        return d

    x = d["Delta_Pseudo"].values
    y = d["Delta_Opt"].values

    med_x = np.nanmedian(x)
    med_y = np.nanmedian(y)

    mad_x = 1.4826 * np.nanmedian(np.abs(x - med_x))
    mad_y = 1.4826 * np.nanmedian(np.abs(y - med_y))

    mad_x = max(mad_x, 0.03)
    mad_y = max(mad_y, 0.03)

    keep = (
        (np.abs(x - med_x) < 4.0 * mad_x)
        & (np.abs(y - med_y) < 4.0 * mad_y)
    )

    d = d.loc[keep].copy()

    # Optional: prefer better photometry.
    if (
        "e_Delta_Pseudo" in d.columns
        and "e_Delta_Opt" in d.columns
        and len(d) > 100
    ):
        ep_lim = np.nanpercentile(d["e_Delta_Pseudo"], 90)
        eo_lim = np.nanpercentile(d["e_Delta_Opt"], 90)

        d = d[
            (d["e_Delta_Pseudo"] <= ep_lim)
            & (d["e_Delta_Opt"] <= eo_lim)
        ].copy()

    return d


def build_all_templates(
    data: pd.DataFrame,
    fiducials: dict,
    cfg: ClusterConfig,
):
    """
    Build physical templates:
    - single_2g
    - single_1g
    - bin_2g2g
    - bin_1g1g
    - bin_1g2g

    For mock catalogs, if pop_truth is available, this function uses
    truth 1G/2G labels by default. This avoids severe template bias from
    applying real-data chromosome-map cuts to simulated catalogs.
    """
    used_truth_templates = False

    # ------------------------------------------------------------
    # Mock path: use truth population labels if available.
    # ------------------------------------------------------------
    use_truth_pop_for_templates = getattr(
        cfg,
        "use_truth_pop_for_templates",
        "pop_truth" in data.columns,
    )

    if use_truth_pop_for_templates and "pop_truth" in data.columns:
        clean_1g_truth, clean_2g_truth = _select_truth_clean_populations(
            data,
            cfg,
        )

        if clean_1g_truth is not None and clean_2g_truth is not None:
            clean_1g = clean_1g_truth
            clean_2g = clean_2g_truth
            used_truth_templates = True

            print(
                "Using mock truth labels for template construction: "
                f"N1G={len(clean_1g)}, N2G={len(clean_2g)}"
            )
        else:
            print(
                "pop_truth column found, but could not identify usable "
                "truth 1G/2G template samples. Falling back to "
                "chromosome-map template selection."
            )

    # ------------------------------------------------------------
    # Real-data/default path: original chromosome-map/GMM selection.
    # ------------------------------------------------------------
    if not used_truth_templates:
        print("Using chromosome-map selection for template construction.")

        clean_1g = robust_clean_population(
            data,
            pop_id=1,
            cfg=cfg,
            prob_col="p_init_1g",
            min_prob=cfg.init_prob_threshold,
        )

        clean_2g = robust_clean_population(
            data,
            pop_id=2,
            cfg=cfg,
            prob_col="p_init_2g",
            min_prob=cfg.init_prob_threshold,
        )

        print(f"Clean 1G template stars: {len(clean_1g)}")
        print(f"Clean 2G template stars: {len(clean_2g)}")

    # ------------------------------------------------------------
    # Minimum sample-size check.
    # ------------------------------------------------------------
    min_template_stars = getattr(
        cfg,
        "min_template_stars",
        50,
    )

    if len(clean_1g) < min_template_stars or len(clean_2g) < min_template_stars:
        raise RuntimeError(
            f"Too few clean 1G/2G stars for templates: "
            f"N1G={len(clean_1g)}, N2G={len(clean_2g)}, "
            f"min_template_stars={min_template_stars}"
        )

    pts_single_1g = clean_1g[["Delta_Pseudo", "Delta_Opt"]].values
    pts_single_2g = clean_2g[["Delta_Pseudo", "Delta_Opt"]].values

    pts_bin_1g1g = generate_binary_template(
        clean_1g,
        clean_1g,
        fiducials,
        cfg,
        cfg.n_binary_template,
    )

    pts_bin_2g2g = generate_binary_template(
        clean_2g,
        clean_2g,
        fiducials,
        cfg,
        cfg.n_binary_template,
    )

    # Mixed binaries. We generate both directions and combine.
    pts_bin_1g2g_a = generate_binary_template(
        clean_1g,
        clean_2g,
        fiducials,
        cfg,
        cfg.n_binary_template // 2,
    )

    pts_bin_1g2g_b = generate_binary_template(
        clean_2g,
        clean_1g,
        fiducials,
        cfg,
        cfg.n_binary_template // 2,
    )

    pts_bin_1g2g = np.vstack([pts_bin_1g2g_a, pts_bin_1g2g_b])

    template_points = {
        "single_2g": pts_single_2g,
        "single_1g": pts_single_1g,
        "bin_2g2g": pts_bin_2g2g,
        "bin_1g1g": pts_bin_1g1g,
        "bin_1g2g": pts_bin_1g2g,
    }

    template_gmms = {
        "single_2g": fit_template_gmm(
            pts_single_2g,
            cfg.gmm_components_single,
            cfg,
        ),
        "single_1g": fit_template_gmm(
            pts_single_1g,
            cfg.gmm_components_single,
            cfg,
        ),
        "bin_2g2g": fit_template_gmm(
            pts_bin_2g2g,
            cfg.gmm_components_binary,
            cfg,
        ),
        "bin_1g1g": fit_template_gmm(
            pts_bin_1g1g,
            cfg.gmm_components_binary,
            cfg,
        ),
        "bin_1g2g": fit_template_gmm(
            pts_bin_1g2g,
            cfg.gmm_components_binary,
            cfg,
        ),
    }

    return template_points, template_gmms
