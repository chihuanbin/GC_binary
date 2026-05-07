# gc_binary_pipeline/run_cluster.py

import os
import json
import pandas as pd
import numpy as np

from .config import ClusterConfig
from .io_hugs import load_hugs_catalog, quality_selection, downsample_if_needed
from .differential_reddening import apply_differential_reddening_correction
from .chromosome_map import (
    construct_chromosome_map,
    initial_population_labels,
    add_colors,
    normalize_two_fiducial,
)
from .photometric_errors import compute_chromosome_errors
from .binary_forward_model import (
    build_all_templates,
    build_field_template,
    fit_template_gmm,
)
from .mixture_model import run_template_bayesian_model, summarize_trace
from .plotting import (
    ensure_dir,
    plot_cmd,
    plot_chromosome_map,
    plot_templates,
    plot_photometric_errors,
    plot_corner,
)


def transform_all_for_field(df_all, fiducials):
    """
    Transform the full catalog into Delta coordinates for empirical field template.

    This is intentionally simple. It uses corrected/raw magnitudes available
    in df_all. In a more rigorous implementation, apply the same DR correction
    to all stars before this step.
    """
    d = df_all.copy()

    needed = ["F275W", "F336W", "F435W", "F606W", "F814W"]
    for col in needed:
        if col not in d.columns:
            raise KeyError(f"Missing column {col}")

    d = add_colors(d)

    mag = d["F606W"].values

    d["Delta_Pseudo"] = normalize_two_fiducial(
        mag,
        d["C_pseudo"].values,
        fiducials["pseudo_blue"],
        fiducials["pseudo_red"],
        center=True,
    )

    d["Delta_Opt"] = normalize_two_fiducial(
        mag,
        d["Color_Opt"].values,
        fiducials["opt_blue"],
        fiducials["opt_red"],
        center=True,
    )

    finite = np.isfinite(d["Delta_Pseudo"].values) & np.isfinite(d["Delta_Opt"].values)
    return d.loc[finite].copy()


def fallback_field_template(data, cfg):
    """
    Fallback broad field component if low-membership stars are unavailable.
    """
    pts = data[["Delta_Pseudo", "Delta_Opt"]].values

    # Inflate observed distribution by adding jitter.
    rng = np.random.default_rng(cfg.random_seed)
    idx = rng.choice(len(pts), size=min(10000, len(pts)), replace=True)
    field_pts = pts[idx].copy()
    field_pts += rng.normal(0, 0.25, size=field_pts.shape)

    return fit_template_gmm(
        field_pts,
        n_components=cfg.gmm_components_field,
        cfg=cfg,
    )


def run_cluster(cfg: ClusterConfig):
    """
    Full end-to-end pipeline for one cluster.
    """
    outdir = os.path.join(cfg.output_dir, cfg.cluster_name)
    ensure_dir(outdir)

    print(f"\n=== Running cluster: {cfg.cluster_name} ===")
    print(f"Input: {cfg.file_path}")

    # 1. Load
    df_all = load_hugs_catalog(cfg.file_path)

    # 2. Quality selection
    data = quality_selection(df_all, cfg)
    print(f"After quality selection: N={len(data)}")

    # 3. Differential reddening correction
    if cfg.apply_dr:
        print("Applying local differential reddening correction...")
        do_dr = getattr(cfg, "apply_dr", True)

        has_xy = (
            "X" in data.columns
            and "Y" in data.columns
            and data["X"].notna().any()
            and data["Y"].notna().any()
        )

        if do_dr and has_xy:
            print("Applying local differential reddening correction...")
            data = apply_differential_reddening_correction(data, cfg)
        elif do_dr and not has_xy:
            print("Skipping differential reddening correction: x/y positions unavailable.")
        else:
            print("Skipping differential reddening correction: disabled by config.")

    # 4. Diagnostic CMD
    plot_cmd(data, cfg, outdir)

    # 5. Construct two-fiducial chromosome-map-like coordinates
    data, fiducials = construct_chromosome_map(data, cfg)
    print(f"After chromosome-map finite selection: N={len(data)}")

    # 6. Photometric errors in Delta coordinates
    data = compute_chromosome_errors(data, fiducials, cfg)

    # 7. Downsample after all transformations
    data = downsample_if_needed(data, cfg)
    print(f"After optional downsampling: N={len(data)}")

    # 8. Initial population labels for templates
    data = initial_population_labels(data, cfg)

    # Diagnostics
    plot_chromosome_map(data, cfg, outdir, color_by_init=True)
    plot_photometric_errors(data, cfg, outdir)

    # 9. Build templates
    print("Building empirical/forward binary templates...")
    template_points, template_gmms = build_all_templates(data, fiducials, cfg)

    # 10. Field template
    print("Building field template...")
    try:
        df_all_map = transform_all_for_field(df_all, fiducials)
        field_gmm = build_field_template(df_all_map, cfg)
    except Exception as e:
        print(f"Field template from low-membership stars failed: {e}")
        field_gmm = None

    if field_gmm is None:
        print("Using fallback broad field template.")
        field_gmm = fallback_field_template(data, cfg)

    template_gmms["field"] = field_gmm

    plot_templates(template_points, data, cfg, outdir)

    # 11. Bayesian model
    print("Running Bayesian mixture model...")
    trace = run_template_bayesian_model(data, template_gmms, cfg)

    # 12. Summary
    summary = summarize_trace(trace)
    summary["cluster"] = cfg.cluster_name
    summary["N_used"] = len(data)
    summary["mag_min"] = cfg.mag_min
    summary["mag_max"] = cfg.mag_max
    summary["apply_dr"] = cfg.apply_dr

    summary_path = os.path.join(outdir, f"{cfg.cluster_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(
        os.path.join(outdir, f"{cfg.cluster_name}_summary.csv"),
        index=False,
    )

    # 13. Save transformed data
    data.to_csv(
        os.path.join(outdir, f"{cfg.cluster_name}_processed_data.csv"),
        index=False,
    )

    # 14. Save posterior
    trace.to_netcdf(os.path.join(outdir, f"{cfg.cluster_name}_trace.nc"))

    # 15. Corner plot
    plot_corner(trace, cfg, outdir)

    print("Finished.")
    print(json.dumps(summary, indent=2))

    return {
        "data": data,
        "fiducials": fiducials,
        "template_points": template_points,
        "template_gmms": template_gmms,
        "trace": trace,
        "summary": summary,
    }


if __name__ == "__main__":
    # Example for NGC5272.
    cfg = ClusterConfig(
        cluster_name="NGC5272",
        file_path="golden_samples/HST_56GC/ngc5272/hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc5272_multi_v1_catalog-meth1.txt",
        mag_min=19.0,
        mag_max=21.5,
        sample_size=8000,
        apply_dr=True,
        output_dir="outputs",
    )

    run_cluster(cfg)
