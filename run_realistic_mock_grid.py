#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_realistic_mock_grid.py

Generate and run a grid of realistic Milone-like mock catalogues through the
same gc_binary_pipeline used for real clusters.

Outputs:
  outputs_realistic_mock/
      realistic_mock_grid_summary.csv
      mocks/<scenario>/<mock_name>_catalog.txt
      mocks/<scenario>/<mock_name>_metadata.json
      pipeline_outputs/...
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import logging
import traceback
from dataclasses import fields, is_dataclass
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

from generate_realistic_mock import generate_realistic_mock

from gc_binary_pipeline.config import ClusterConfig
from gc_binary_pipeline.run_cluster import run_cluster


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_scalar(x):
    return isinstance(
        x,
        (
            int,
            float,
            str,
            bool,
            np.integer,
            np.floating,
            type(None),
        ),
    )


def extract_scalar_summary(result):
    """
    Extract scalar summary from run_cluster result.
    """
    if result is None:
        return {}

    if not isinstance(result, dict):
        return {"result_repr": repr(result)}

    if "summary" in result and isinstance(result["summary"], dict):
        src = result["summary"]
    else:
        src = result

    out = {}
    for k, v in src.items():
        if is_scalar(v):
            out[k] = v
    return out


def make_cluster_config_safe(**kwargs):
    """
    Construct ClusterConfig while ignoring keys not accepted by the current
    dataclass definition.

    This makes the script robust to small config differences.
    """
    if is_dataclass(ClusterConfig):
        allowed = {f.name for f in fields(ClusterConfig)}
        clean = {k: v for k, v in kwargs.items() if k in allowed}
        ignored = sorted(set(kwargs) - set(clean))
        if ignored:
            logging.debug(f"Ignored unsupported ClusterConfig keys: {ignored}")
        return ClusterConfig(**clean)

    # Fallback if ClusterConfig is not dataclass
    return ClusterConfig(**kwargs)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def classify_recovery(rec):
    """
    Simple QC/recovery classification for mock tests.
    """
    status = str(rec.get("Status", "")).lower()
    if status != "success":
        return "run_failed"

    w_field = safe_float(rec.get("w_field_mean"))
    f_mixed = safe_float(rec.get("f_bin_mixed_mean"))
    n_used = safe_float(rec.get("N_used"))
    delta_hat = safe_float(rec.get("delta_f_bin_2g_minus_1g_mean"))
    delta_true = safe_float(rec.get("true_delta"))

    reasons = []

    if np.isfinite(n_used) and n_used < 800:
        reasons.append("low_N")
    if np.isfinite(w_field) and w_field > 0.10:
        reasons.append("high_field")
    if np.isfinite(f_mixed) and f_mixed > 0.30:
        reasons.append("high_mixed")
    if np.isfinite(delta_hat) and np.isfinite(delta_true):
        if delta_true < 0 and delta_hat < 0:
            reasons.append("sign_correct")
        elif delta_true > 0 and delta_hat > 0:
            reasons.append("sign_correct")
        elif abs(delta_true) < 1e-8:
            reasons.append("equal_case")
        else:
            reasons.append("sign_wrong")

    if not reasons:
        return "ok"

    return ";".join(reasons)


# ============================================================
# Scenario grid
# ============================================================

def build_scenarios():
    """
    Main realistic mock scenarios.

    Keep this grid manageable for first paper revision.
    Increase n_rep in CLI for final production.
    """
    scenarios = [
        {
            "scenario": "equal_binary_fractions",
            "f_bin_1g": 0.05,
            "f_bin_2g": 0.05,
            "f_bin_mixed": 0.01,
            "w_field": 0.03,
            "dr_amp": 0.030,
            "phot_error_scale": 1.0,
            "description": "Equal 1G1G and 2G2G fractions; checks no artificial negative bias.",
        },
        {
            "scenario": "mild_2g_deficit",
            "f_bin_1g": 0.06,
            "f_bin_2g": 0.03,
            "f_bin_mixed": 0.01,
            "w_field": 0.03,
            "dr_amp": 0.030,
            "phot_error_scale": 1.0,
            "description": "Fiducial-like mild 2G2G deficit.",
        },
        {
            "scenario": "strong_2g_deficit",
            "f_bin_1g": 0.10,
            "f_bin_2g": 0.02,
            "f_bin_mixed": 0.01,
            "w_field": 0.03,
            "dr_amp": 0.030,
            "phot_error_scale": 1.0,
            "description": "Strong 2G2G deficit.",
        },
        {
            "scenario": "opposite_sign",
            "f_bin_1g": 0.03,
            "f_bin_2g": 0.07,
            "f_bin_mixed": 0.01,
            "w_field": 0.03,
            "dr_amp": 0.030,
            "phot_error_scale": 1.0,
            "description": "Opposite-sign test; checks pipeline can recover positive delta.",
        },
        {
            "scenario": "high_mixed_component",
            "f_bin_1g": 0.05,
            "f_bin_2g": 0.03,
            "f_bin_mixed": 0.08,
            "w_field": 0.03,
            "dr_amp": 0.030,
            "phot_error_scale": 1.0,
            "description": "Mixed-binary stress test.",
        },
        {
            "scenario": "high_field_contamination",
            "f_bin_1g": 0.05,
            "f_bin_2g": 0.03,
            "f_bin_mixed": 0.01,
            "w_field": 0.15,
            "dr_amp": 0.030,
            "phot_error_scale": 1.0,
            "description": "High-field stress test expected to trigger QC.",
        },
        {
            "scenario": "severe_dr_phot_errors",
            "f_bin_1g": 0.06,
            "f_bin_2g": 0.03,
            "f_bin_mixed": 0.01,
            "w_field": 0.05,
            "dr_amp": 0.060,
            "phot_error_scale": 1.5,
            "description": "Severe DR and photometric-error stress test.",
        },
    ]

    return scenarios


# ============================================================
# Run one mock
# ============================================================

def run_one_mock(
    scenario,
    rep,
    seed,
    base_outdir,
    n_total,
    mag_min,
    mag_max,
    sample_size,
    draws,
    tune,
    chains,
    run_pipeline=True,
):
    scenario_name = scenario["scenario"]

    mock_name = f"{scenario_name}_rep{rep:03d}_seed{seed}"
    mock_dir = ensure_dir(base_outdir / "mocks" / scenario_name)

    logging.info("=" * 80)
    logging.info(f"Generating mock: {mock_name}")

    catalog_path, truth_path, metadata_path, metadata = generate_realistic_mock(
        output_dir=mock_dir,
        mock_name=mock_name,
        seed=seed,
        n_total=n_total,
        mag_min=mag_min,
        mag_max=mag_max,
        f_2g=0.70,
        f_bin_1g=scenario["f_bin_1g"],
        f_bin_2g=scenario["f_bin_2g"],
        f_bin_mixed=scenario["f_bin_mixed"],
        w_field=scenario["w_field"],
        q_mode="flat",
        dr_amp=scenario["dr_amp"],
        phot_error_scale=scenario["phot_error_scale"],
        crowding_outlier_frac=0.01,
        radial_gradient=True,
    )

    rec = {
        "scenario": scenario_name,
        "rep": rep,
        "seed": seed,
        "mock_name": mock_name,
        "Status": "GeneratedOnly" if not run_pipeline else None,
        "catalog_path": str(catalog_path),
        "truth_path": str(truth_path),
        "metadata_path": str(metadata_path),
        "description": scenario.get("description", ""),
    }

    # Add truth metadata
    for k, v in metadata.items():
        rec[f"true_{k}"] = v

    # Short aliases
    rec["true_f_bin_1g1g"] = metadata["realised_f_bin_1g1g"]
    rec["true_f_bin_2g2g"] = metadata["realised_f_bin_2g2g"]
    rec["true_f_bin_mixed"] = metadata["realised_f_bin_mixed"]
    rec["true_delta"] = metadata["realised_delta"]
    rec["true_w_field"] = metadata["realised_w_field"]
    rec["true_w_2g"] = metadata["realised_w_2g_cluster"]

    if not run_pipeline:
        return rec

    pipeline_out = ensure_dir(base_outdir / "pipeline_outputs" / scenario_name / mock_name)

    cfg_kwargs = dict(
        cluster_name=mock_name,
        file_path=str(catalog_path),
        mag_min=mag_min,
        mag_max=mag_max,
        sample_size=sample_size,
        apply_dr=True,
        reverse_population_assignment=False,

        # These are deliberately relaxed enough for complex mocks,
        # while still close to your clean090-style configuration.
        init_prob_threshold=0.90,
        delta_abs_max=3.5,
        template_delta_abs_max=2.0,
        binary_primary_mag_buffer=1.0,

        # Increase if template clipping failures occur
        n_binary_template=20000,

        draws=draws,
        tune=tune,
        chains=chains,
        output_dir=str(pipeline_out),
    )

    cfg = make_cluster_config_safe(**cfg_kwargs)

    logging.info(f"Running pipeline for mock: {mock_name}")
    logging.info(f"catalog_path = {catalog_path}")

    try:
        result = run_cluster(cfg)
        summary = extract_scalar_summary(result)

        rec["Status"] = "Success"
        rec.update(summary)

    except Exception as e:
        rec["Status"] = "Failed"
        rec["Exception"] = repr(e)
        rec["Traceback"] = traceback.format_exc()
        logging.error(f"Pipeline failed for mock: {mock_name}")
        logging.error(traceback.format_exc())

    rec["mock_recovery_flags"] = classify_recovery(rec)
    return rec


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--output-dir", default="outputs_realistic_mock")
    p.add_argument("--n-rep", type=int, default=30)
    p.add_argument("--seed0", type=int, default=202600)

    p.add_argument("--n-total", type=int, default=5000)
    p.add_argument("--mag-min", type=float, default=18.0)
    p.add_argument("--mag-max", type=float, default=21.5)
    p.add_argument("--sample-size", type=int, default=5000)

    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)

    p.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate mocks; do not run gc_binary_pipeline.",
    )

    p.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional subset of scenario names.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    base_outdir = ensure_dir(args.output_dir)
    summary_csv = base_outdir / "realistic_mock_grid_summary.csv"

    scenarios = build_scenarios()

    if args.scenarios is not None:
        keep = set(args.scenarios)
        scenarios = [s for s in scenarios if s["scenario"] in keep]

    logging.info(f"Output directory: {base_outdir}")
    logging.info(f"Number of scenarios: {len(scenarios)}")
    logging.info(f"Repetitions per scenario: {args.n_rep}")
    logging.info(f"Run pipeline: {not args.generate_only}")

    # Save scenario definitions
    scenario_json = base_outdir / "realistic_mock_scenarios.json"
    with open(scenario_json, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)

    records = []

    # If existing partial summary exists, continue from scratch by default.
    # You can modify this section to resume if needed.
    for s_idx, scenario in enumerate(scenarios):
        for rep in range(args.n_rep):
            seed = args.seed0 + 10000 * s_idx + rep

            rec = run_one_mock(
                scenario=scenario,
                rep=rep,
                seed=seed,
                base_outdir=base_outdir,
                n_total=args.n_total,
                mag_min=args.mag_min,
                mag_max=args.mag_max,
                sample_size=args.sample_size,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                run_pipeline=not args.generate_only,
            )

            records.append(rec)

            # Incremental save
            df_tmp = pd.DataFrame(records)
            df_tmp.to_csv(summary_csv, index=False, encoding="utf-8-sig")
            logging.info(f"Saved partial summary: {summary_csv}")

    df = pd.DataFrame(records)

    # Preferred column order
    preferred = [
        "scenario",
        "rep",
        "seed",
        "mock_name",
        "Status",
        "mock_recovery_flags",
        "true_f_bin_1g1g",
        "true_f_bin_2g2g",
        "true_f_bin_mixed",
        "true_delta",
        "true_w_field",
        "true_w_2g",
        "f_bin_1g1g_mean",
        "f_bin_1g1g_sd",
        "f_bin_2g2g_mean",
        "f_bin_2g2g_sd",
        "f_bin_mixed_mean",
        "f_bin_mixed_sd",
        "delta_f_bin_2g_minus_1g_mean",
        "delta_f_bin_2g_minus_1g_sd",
        "delta_f_bin_2g_minus_1g_hdi_2.5",
        "delta_f_bin_2g_minus_1g_hdi_97.5",
        "p_delta_lt0",
        "w_field_mean",
        "w_2g_mean",
        "N_used",
        "catalog_path",
        "metadata_path",
        "Exception",
    ]

    cols = [c for c in preferred if c in df.columns]
    other = [c for c in df.columns if c not in cols]
    df = df[cols + other]

    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    logging.info("=" * 80)
    logging.info(f"Saved final summary: {summary_csv}")

    key_cols = [
        "scenario",
        "rep",
        "Status",
        "true_delta",
        "delta_f_bin_2g_minus_1g_mean",
        "p_delta_lt0",
        "w_field_mean",
        "f_bin_mixed_mean",
        "mock_recovery_flags",
    ]
    key_cols = [c for c in key_cols if c in df.columns]
    logging.info("\n" + df[key_cols].to_string(index=False))


if __name__ == "__main__":
    freeze_support()
    main()
