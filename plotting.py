# gc_binary_pipeline/plotting.py

import os
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import corner

from .mixture_model import CLASS_ORDER


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_cmd(data, cfg, outdir):
    ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.scatter(
        data["F606W"] - data["F814W"],
        data["F606W"],
        s=2,
        c="k",
        alpha=0.25,
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"$F606W-F814W$")
    ax.set_ylabel(r"$F606W$")
    ax.set_title(cfg.cluster_name)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{cfg.cluster_name}_CMD.png"), dpi=250)
    plt.close(fig)


def plot_chromosome_map(data, cfg, outdir, color_by_init=True):
    ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(6, 5))

    if color_by_init and "pop_init" in data.columns:
        colors = {0: "0.7", 1: "tab:blue", 2: "tab:red"}
        labels = {0: "unclassified", 1: "initial 1G", 2: "initial 2G"}

        for k in [0, 1, 2]:
            d = data[data["pop_init"] == k]
            if len(d) == 0:
                continue
            ax.scatter(
                d["Delta_Pseudo"],
                d["Delta_Opt"],
                s=3,
                alpha=0.35,
                c=colors[k],
                label=labels[k],
            )
        ax.legend(markerscale=4, fontsize=8)
    else:
        ax.scatter(
            data["Delta_Pseudo"],
            data["Delta_Opt"],
            s=3,
            alpha=0.35,
            c="k",
        )

    ax.set_xlabel(r"$\Delta_{\rm pseudo}$")
    ax.set_ylabel(r"$\Delta_{\rm opt}$")
    ax.set_title(f"{cfg.cluster_name}: two-fiducial map")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"{cfg.cluster_name}_chromosome_map.png"),
        dpi=250,
    )
    plt.close(fig)


def plot_templates(template_points, data, cfg, outdir):
    ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        data["Delta_Pseudo"],
        data["Delta_Opt"],
        s=2,
        c="0.8",
        alpha=0.25,
        label="observed",
    )

    colors = {
        "single_2g": "tab:red",
        "single_1g": "tab:blue",
        "bin_2g2g": "darkred",
        "bin_1g1g": "navy",
        "bin_1g2g": "tab:green",
    }

    for name, pts in template_points.items():
        if pts is None or len(pts) == 0:
            continue
        idx = np.random.choice(
            len(pts),
            size=min(3000, len(pts)),
            replace=False,
        )
        ax.scatter(
            pts[idx, 0],
            pts[idx, 1],
            s=3,
            alpha=0.25,
            c=colors.get(name, "k"),
            label=name,
        )

    ax.set_xlabel(r"$\Delta_{\rm pseudo}$")
    ax.set_ylabel(r"$\Delta_{\rm opt}$")
    ax.set_title(f"{cfg.cluster_name}: empirical binary templates")
    ax.legend(markerscale=4, fontsize=8)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"{cfg.cluster_name}_binary_templates.png"),
        dpi=250,
    )
    plt.close(fig)


def plot_photometric_errors(data, cfg, outdir):
    ensure_dir(outdir)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].scatter(
        data["F606W"],
        data["e_Delta_Pseudo"],
        s=3,
        alpha=0.3,
        c="tab:purple",
    )
    ax[0].set_xlabel(r"$F606W$")
    ax[0].set_ylabel(r"$\sigma(\Delta_{\rm pseudo})$")

    ax[1].scatter(
        data["F606W"],
        data["e_Delta_Opt"],
        s=3,
        alpha=0.3,
        c="tab:orange",
    )
    ax[1].set_xlabel(r"$F606W$")
    ax[1].set_ylabel(r"$\sigma(\Delta_{\rm opt})$")

    fig.suptitle(f"{cfg.cluster_name}: propagated errors")
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"{cfg.cluster_name}_photometric_errors.png"),
        dpi=250,
    )
    plt.close(fig)


def plot_corner(trace, cfg, outdir):
    ensure_dir(outdir)

    var_names = [
        "f_bin_1g1g",
        "f_bin_2g2g",
        "f_bin_mixed",
        "ratio_pure_2g2g_1g1g",
        "ratio_effective",
        "w_field",
    ]

    ds = az.extract(trace, var_names=var_names)
    samples = ds.to_dataframe().reset_index(drop=True)

    labels = [
        r"$f_{\rm bin}^{1G-1G}$",
        r"$f_{\rm bin}^{2G-2G}$",
        r"$f_{\rm bin}^{1G-2G}$",
        r"$R_{\rm pure}$",
        r"$R_{\rm eff}$",
        r"$w_{\rm field}$",
    ]

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        color="#0055AA",
        plot_density=True,
        plot_contours=True,
    )

    fig.suptitle(f"{cfg.cluster_name}: posterior", fontsize=14)
    fig.savefig(
        os.path.join(outdir, f"{cfg.cluster_name}_corner.png"),
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)
