# gc_binary_pipeline/mixture_model.py

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from .config import ClusterConfig


CLASS_ORDER = [
    "single_2g",
    "single_1g",
    "bin_2g2g",
    "bin_1g1g",
    "bin_1g2g",
    "field",
]


def prepare_template_arrays(template_gmms: dict):
    """
    Flatten class-level template GMMs into subcomponents.
    """
    all_mu = []
    all_varx = []
    all_vary = []
    all_subw = []
    class_id = []

    for j, name in enumerate(CLASS_ORDER):
        if name not in template_gmms:
            raise KeyError(f"Missing template GMM for class: {name}")

        g = template_gmms[name]
        w = np.asarray(g["weights"])
        mu = np.asarray(g["means"])
        cov = np.asarray(g["covs"])

        K = len(w)

        all_mu.append(mu)
        all_varx.append(cov[:, 0, 0])
        all_vary.append(cov[:, 1, 1])
        all_subw.append(w)
        class_id.extend([j] * K)

    return {
        "mu": np.vstack(all_mu).astype(float),
        "varx": np.concatenate(all_varx).astype(float),
        "vary": np.concatenate(all_vary).astype(float),
        "subw": np.concatenate(all_subw).astype(float),
        "class_id": np.asarray(class_id, dtype="int32"),
        "class_names": CLASS_ORDER,
    }


def logsumexp_pt(a, axis=None):
    amax = pt.max(a, axis=axis, keepdims=True)
    out = pt.log(pt.sum(pt.exp(a - amax), axis=axis, keepdims=True)) + amax
    if axis is not None:
        out = pt.squeeze(out, axis=axis)
    return out


def logp_diag_gaussian_2d(x, mu, varx, vary):
    """
    Diagonal Gaussian log-probability in 2D.

    x    : N x 2
    mu   : K x 2
    varx : N x K
    vary : N x K

    returns N x K
    """
    dx = x[:, None, 0] - mu[None, :, 0]
    dy = x[:, None, 1] - mu[None, :, 1]

    logdet = pt.log(varx) + pt.log(vary)
    quad = dx**2 / varx + dy**2 / vary

    return -0.5 * (2.0 * np.log(2.0 * np.pi) + logdet + quad)


def run_template_bayesian_model(
    data: pd.DataFrame,
    template_gmms: dict,
    cfg: ClusterConfig,
):
    """
    Bayesian mixture model with:
    - template-GMM morphology;
    - explicit 1G--1G, 2G--2G, and mixed 1G--2G binaries;
    - field contamination;
    - star-by-star photometric uncertainty convolution.
    """
    obs = data[["Delta_Pseudo", "Delta_Opt"]].values.astype(float)
    ex = data["e_Delta_Pseudo"].values.astype(float)
    ey = data["e_Delta_Opt"].values.astype(float)

    tpl = prepare_template_arrays(template_gmms)

    with pm.Model() as model:
        # Data containers
        x_obs = pm.Data("x_obs", obs)
        ex_obs = pm.Data("ex_obs", ex)
        ey_obs = pm.Data("ey_obs", ey)

        mu_tpl = pm.Data("mu_tpl", tpl["mu"])
        varx_tpl = pm.Data("varx_tpl", tpl["varx"])
        vary_tpl = pm.Data("vary_tpl", tpl["vary"])
        subw_tpl = pm.Data("subw_tpl", tpl["subw"])
        class_id = pm.Data("class_id", tpl["class_id"])

        # -----------------------------
        # Physical parameters
        # -----------------------------
        w_field = pm.Beta("w_field", alpha=1.0, beta=20.0)

        # 2G population fraction among cluster stars
        w_2g = pm.Beta("w_2g", alpha=5.0, beta=3.0)

        # Same-population binary fractions
        f_bin_1g1g = pm.Beta("f_bin_1g1g", alpha=1.5, beta=8.5)
        f_bin_2g2g = pm.Beta("f_bin_2g2g", alpha=1.5, beta=8.5)

        # Mixed binary fraction as nuisance component
        f_bin_mixed = pm.Beta("f_bin_mixed", alpha=1.0, beta=10.0)

        # Cluster-internal raw weights
        raw_cluster = pt.stack([
            w_2g * (1.0 - f_bin_2g2g),          # single_2g
            (1.0 - w_2g) * (1.0 - f_bin_1g1g),  # single_1g
            w_2g * f_bin_2g2g,                  # bin_2g2g
            (1.0 - w_2g) * f_bin_1g1g,          # bin_1g1g
            f_bin_mixed,                        # bin_1g2g
        ])

        raw_cluster = raw_cluster / pt.sum(raw_cluster)

        class_weights = pt.concatenate([
            (1.0 - w_field) * raw_cluster,
            pt.stack([w_field]),
        ])

        # Subcomponent weights
        comp_w = class_weights[class_id] * subw_tpl
        comp_w = comp_w / pt.sum(comp_w)

        # Star-by-star measurement error convolution
        varx = varx_tpl[None, :] + ex_obs[:, None] ** 2
        vary = vary_tpl[None, :] + ey_obs[:, None] ** 2

        logp_comp = logp_diag_gaussian_2d(
            x_obs,
            mu_tpl,
            varx,
            vary,
        )

        logp_comp = logp_comp + pt.log(comp_w[None, :] + 1e-30)
        logp_star = logsumexp_pt(logp_comp, axis=1)

        pm.Potential("likelihood", pt.sum(logp_star))

        # Ratios
        ratio_pure = pm.Deterministic(
            "ratio_pure_2g2g_1g1g",
            f_bin_2g2g / pt.maximum(f_bin_1g1g, 1e-4),
        )

        ratio_eff = pm.Deterministic(
            "ratio_effective",
            (f_bin_2g2g + 0.5 * f_bin_mixed)
            / pt.maximum(f_bin_1g1g + 0.5 * f_bin_mixed, 1e-4),
        )
        delta_f_bin = pm.Deterministic(
        "delta_f_bin_2g_minus_1g",
        f_bin_2g2g - f_bin_1g1g,
        )


        trace = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed,
            return_inferencedata=True,
        )

    return trace


def summarize_trace(trace) -> dict:
    """
    Extract key posterior summaries.
    """
    var_names = [
        "f_bin_1g1g",
        "f_bin_2g2g",
        "f_bin_mixed",
        "ratio_pure_2g2g_1g1g",
        "ratio_effective",
        "w_field",
        "w_2g",
        "delta_f_bin_2g_minus_1g",

    ]

    summary = az.summary(
        trace,
        var_names=var_names,
        hdi_prob=0.95,
    )

    out = {}
    for name in var_names:
        if name in summary.index:
            out[f"{name}_mean"] = float(summary.loc[name, "mean"])
            out[f"{name}_sd"] = float(summary.loc[name, "sd"])
            out[f"{name}_hdi_2.5"] = float(summary.loc[name, "hdi_2.5%"])
            out[f"{name}_hdi_97.5"] = float(summary.loc[name, "hdi_97.5%"])

    return out
