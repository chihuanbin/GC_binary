# gc_binary_pipeline/injection_tests.py

import numpy as np
import pandas as pd

from .mixture_model import CLASS_ORDER, run_template_bayesian_model, summarize_trace
from .config import ClusterConfig


def draw_from_gmm(gmm, n, rng):
    weights = np.asarray(gmm["weights"])
    means = np.asarray(gmm["means"])
    covs = np.asarray(gmm["covs"])

    comp = rng.choice(len(weights), size=n, p=weights)

    pts = np.zeros((n, 2))
    for k in range(len(weights)):
        m = comp == k
        if np.any(m):
            pts[m] = rng.multivariate_normal(
                means[k],
                covs[k],
                size=np.sum(m),
            )

    return pts


def simulate_realistic_cluster(
    template_gmms: dict,
    cfg: ClusterConfig,
    n_star: int = 5000,
    w_2g: float = 0.65,
    f_bin_1g1g: float = 0.10,
    f_bin_2g2g: float = 0.05,
    f_bin_mixed: float = 0.02,
    w_field: float = 0.03,
    error_median: float = 0.025,
    error_sigma_ln: float = 0.35,
    dr_sigma: float = 0.015,
):
    """
    Realistic injection mock based on empirical templates.

    This test is not a proof of uniqueness, but quantifies recovery under
    controlled model complexity.
    """
    rng = np.random.default_rng(cfg.random_seed)

    raw_cluster = np.array([
        w_2g * (1.0 - f_bin_2g2g),
        (1.0 - w_2g) * (1.0 - f_bin_1g1g),
        w_2g * f_bin_2g2g,
        (1.0 - w_2g) * f_bin_1g1g,
        f_bin_mixed,
    ])

    raw_cluster = raw_cluster / raw_cluster.sum()

    class_probs = np.concatenate([
        (1.0 - w_field) * raw_cluster,
        [w_field],
    ])

    class_probs = class_probs / class_probs.sum()

    cls = rng.choice(len(CLASS_ORDER), size=n_star, p=class_probs)

    points = np.zeros((n_star, 2))
    labels = []

    for j, name in enumerate(CLASS_ORDER):
        m = cls == j
        if not np.any(m):
            continue

        pts = draw_from_gmm(template_gmms[name], np.sum(m), rng)

        # residual DR-like broadening
        pts += rng.normal(0.0, dr_sigma, size=pts.shape)

        points[m] = pts
        labels.extend([name] * np.sum(m))

    # Star-by-star photometric errors
    e1 = rng.lognormal(
        mean=np.log(error_median),
        sigma=error_sigma_ln,
        size=n_star,
    )

    e2 = rng.lognormal(
        mean=np.log(error_median),
        sigma=error_sigma_ln,
        size=n_star,
    )

    obs = points.copy()
    obs[:, 0] += rng.normal(0.0, e1)
    obs[:, 1] += rng.normal(0.0, e2)

    mock = pd.DataFrame({
        "Delta_Pseudo": obs[:, 0],
        "Delta_Opt": obs[:, 1],
        "e_Delta_Pseudo": e1,
        "e_Delta_Opt": e2,
        "true_class_index": cls,
    })

    mock["true_class"] = [CLASS_ORDER[i] for i in cls]

    truth = {
        "w_2g": w_2g,
        "f_bin_1g1g": f_bin_1g1g,
        "f_bin_2g2g": f_bin_2g2g,
        "f_bin_mixed": f_bin_mixed,
        "w_field": w_field,
        "ratio_pure": f_bin_2g2g / max(f_bin_1g1g, 1e-4),
        "ratio_effective": (
            f_bin_2g2g + 0.5 * f_bin_mixed
        ) / max(f_bin_1g1g + 0.5 * f_bin_mixed, 1e-4),
    }

    return mock, truth


def run_injection_recovery_suite(
    template_gmms: dict,
    cfg: ClusterConfig,
):
    """
    Small grid of injection-recovery tests.
    """
    configs = [
        dict(f_bin_1g1g=0.10, f_bin_2g2g=0.10, f_bin_mixed=0.00),
        dict(f_bin_1g1g=0.10, f_bin_2g2g=0.05, f_bin_mixed=0.02),
        dict(f_bin_1g1g=0.10, f_bin_2g2g=0.02, f_bin_mixed=0.05),
        dict(f_bin_1g1g=0.20, f_bin_2g2g=0.02, f_bin_mixed=0.05),
    ]

    rows = []

    for i, kw in enumerate(configs):
        mock, truth = simulate_realistic_cluster(
            template_gmms,
            cfg,
            **kw,
        )

        trace = run_template_bayesian_model(
            mock,
            template_gmms,
            cfg,
        )

        rec = summarize_trace(trace)

        row = {"test_id": i}
        for k, v in truth.items():
            row[f"true_{k}"] = v
        row.update(rec)

        rows.append(row)

    return pd.DataFrame(rows)
