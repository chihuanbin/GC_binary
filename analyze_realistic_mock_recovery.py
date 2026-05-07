#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_realistic_mock_recovery.py

Analyze realistic mock injection--recovery results.

Metrics:
  - bias(delta)
  - RMSE(delta)
  - median error
  - 68% scatter
  - 95% HDI coverage
  - sign recovery rate
  - P(delta_hat < 0) for equal-fraction mocks
  - QC fail rate / run fail rate

Input:
  outputs_realistic_mock/realistic_mock_grid_summary.csv

Outputs:
  realistic_mock_recovery_by_scenario.csv
  realistic_mock_recovery_table.tex
  Fig_realistic_mock_delta_recovery.pdf/png
  Fig_realistic_mock_bias_by_scenario.pdf/png
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def sign_accuracy(delta_true, delta_hat, eps=1e-8):
    """
    For non-zero truth, return boolean sign correctness.
    Equal truth is undefined.
    """
    if not np.isfinite(delta_true) or not np.isfinite(delta_hat):
        return np.nan
    if abs(delta_true) <= eps:
        return np.nan
    return np.sign(delta_true) == np.sign(delta_hat)


def coverage_95(delta_true, lo, hi):
    if not np.isfinite(delta_true) or not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    return (lo <= delta_true) and (delta_true <= hi)


def binomial_se(p, n):
    if n <= 0 or not np.isfinite(p):
        return np.nan
    return np.sqrt(p * (1 - p) / n)


def format_float(x, nd=4):
    if not np.isfinite(x):
        return ""
    return f"{x:.{nd}f}"


# ============================================================
# Main analysis
# ============================================================

def compute_recovery_table(df):
    """
    Compute per-scenario recovery statistics.
    """
    rows = []

    for scenario, g in df.groupby("scenario"):
        n_total = len(g)

        success = g["Status"].astype(str).str.lower() == "success"
        gs = g[success].copy()
        n_success = len(gs)
        n_failed = n_total - n_success

        delta_true = gs["true_delta"].to_numpy(dtype=float)
        delta_hat = gs["delta_f_bin_2g_minus_1g_mean"].to_numpy(dtype=float)
        delta_sd = gs.get(
            "delta_f_bin_2g_minus_1g_sd",
            pd.Series(np.nan, index=gs.index),
        ).to_numpy(dtype=float)

        lo = gs.get(
            "delta_f_bin_2g_minus_1g_hdi_2.5",
            pd.Series(np.nan, index=gs.index),
        ).to_numpy(dtype=float)

        hi = gs.get(
            "delta_f_bin_2g_minus_1g_hdi_97.5",
            pd.Series(np.nan, index=gs.index),
        ).to_numpy(dtype=float)

        # If HDI missing, use approximate normal 95 interval
        missing_hdi = ~np.isfinite(lo) | ~np.isfinite(hi)
        lo[missing_hdi] = delta_hat[missing_hdi] - 1.96 * delta_sd[missing_hdi]
        hi[missing_hdi] = delta_hat[missing_hdi] + 1.96 * delta_sd[missing_hdi]

        err = delta_hat - delta_true
        valid = np.isfinite(err)

        n_valid = int(valid.sum())

        bias = np.nanmean(err)
        median_err = np.nanmedian(err)
        rmse = np.sqrt(np.nanmean(err**2))
        scatter68 = 0.5 * (
            np.nanpercentile(err, 84) - np.nanpercentile(err, 16)
        ) if n_valid > 1 else np.nan

        cover = np.array(
            [coverage_95(t, l, h) for t, l, h in zip(delta_true, lo, hi)],
            dtype=float,
        )
        coverage = np.nanmean(cover)
        n_coverage = np.sum(np.isfinite(cover))

        sign_ok = np.array(
            [sign_accuracy(t, h) for t, h in zip(delta_true, delta_hat)],
            dtype=float,
        )
        sign_rate = np.nanmean(sign_ok)
        n_sign = np.sum(np.isfinite(sign_ok))

        # Equal-case diagnostic
        equal_case = np.isfinite(delta_true) & (np.abs(delta_true) <= 0.010)
        if np.any(equal_case):
            p_hat_negative = np.nanmean(delta_hat[equal_case] < 0)
            n_equal = int(np.sum(equal_case))
        else:
            p_hat_negative = np.nan
            n_equal = 0

        # QC fail heuristics
        w_field = gs.get("w_field_mean", pd.Series(np.nan, index=gs.index)).to_numpy(float)
        f_mixed = gs.get("f_bin_mixed_mean", pd.Series(np.nan, index=gs.index)).to_numpy(float)
        n_used = gs.get("N_used", pd.Series(np.nan, index=gs.index)).to_numpy(float)

        qc_fail = (
            (np.isfinite(w_field) & (w_field > 0.10))
            | (np.isfinite(f_mixed) & (f_mixed > 0.30))
            | (np.isfinite(n_used) & (n_used < 800))
        )

        qc_fail_rate = np.nanmean(qc_fail) if n_success > 0 else np.nan

        row = {
            "scenario": scenario,
            "N_total": n_total,
            "N_success": n_success,
            "N_failed": n_failed,
            "run_fail_rate": n_failed / n_total if n_total else np.nan,
            "N_valid_delta": n_valid,
            "true_delta_mean": np.nanmean(delta_true),
            "delta_hat_mean": np.nanmean(delta_hat),
            "bias_delta": bias,
            "median_error_delta": median_err,
            "rmse_delta": rmse,
            "scatter68_error_delta": scatter68,
            "coverage95": coverage,
            "N_coverage": int(n_coverage),
            "sign_recovery_rate": sign_rate,
            "N_sign": int(n_sign),
            "P_delta_hat_lt0_equal_case": p_hat_negative,
            "N_equal_case": n_equal,
            "qc_fail_rate_successful_runs": qc_fail_rate,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def plot_delta_recovery(df, outdir):
    setup_matplotlib()

    success = df["Status"].astype(str).str.lower() == "success"
    d = df[success].copy()

    d = d[
        np.isfinite(d["true_delta"])
        & np.isfinite(d["delta_f_bin_2g_minus_1g_mean"])
    ].copy()

    if len(d) == 0:
        print("No successful delta recovery rows to plot.")
        return

    scenarios = list(d["scenario"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    color_map = {s: cmap(i % 10) for i, s in enumerate(scenarios)}

    fig, ax = plt.subplots(figsize=(6.2, 5.5))

    for s in scenarios:
        g = d[d["scenario"] == s]
        ax.scatter(
            g["true_delta"],
            g["delta_f_bin_2g_minus_1g_mean"],
            s=34,
            alpha=0.75,
            color=color_map[s],
            edgecolor="k",
            linewidth=0.3,
            label=s,
        )

    allv = np.concatenate(
        [
            d["true_delta"].to_numpy(float),
            d["delta_f_bin_2g_minus_1g_mean"].to_numpy(float),
        ]
    )
    lo = np.nanmin(allv) - 0.02
    hi = np.nanmax(allv) + 0.02

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.axhline(0, color="0.4", ls=":", lw=1.0)
    ax.axvline(0, color="0.4", ls=":", lw=1.0)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(r"Input $\Delta f_{\rm bin}$")
    ax.set_ylabel(r"Recovered $\Delta f_{\rm bin}$")

    ax.legend(
        fontsize=8,
        frameon=True,
        loc="best",
        title="Scenario",
        title_fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(outdir / "Fig_realistic_mock_delta_recovery.pdf")
    fig.savefig(outdir / "Fig_realistic_mock_delta_recovery.png", dpi=250)
    plt.close(fig)


def plot_bias_by_scenario(summary, outdir):
    setup_matplotlib()

    d = summary.copy()
    d = d.sort_values("bias_delta")

    x = np.arange(len(d))
    y = d["bias_delta"].to_numpy(float)
    rmse = d["rmse_delta"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    ax.axhline(0, color="k", ls="--", lw=1.1)

    ax.bar(
        x,
        y,
        color="#0072B2",
        alpha=0.75,
        edgecolor="k",
        linewidth=0.5,
        label="Bias",
    )

    ax.errorbar(
        x,
        y,
        yerr=rmse,
        fmt="none",
        ecolor="0.25",
        elinewidth=1.0,
        capsize=3,
        label="RMSE",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(d["scenario"], rotation=45, ha="right", fontsize=9)

    ax.set_ylabel(r"Recovery error in $\Delta f_{\rm bin}$")
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "Fig_realistic_mock_bias_by_scenario.pdf")
    fig.savefig(outdir / "Fig_realistic_mock_bias_by_scenario.png", dpi=250)
    plt.close(fig)


def plot_equal_case_hist(df, outdir):
    setup_matplotlib()

    success = df["Status"].astype(str).str.lower() == "success"
    d = df[success].copy()

    d = d[
        np.isfinite(d["true_delta"])
        & (np.abs(d["true_delta"]) <= 0.010)
        & np.isfinite(d["delta_f_bin_2g_minus_1g_mean"])
    ].copy()

    if len(d) == 0:
        print("No equal-case rows to plot.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.5))

    ax.hist(
        d["delta_f_bin_2g_minus_1g_mean"],
        bins=18,
        color="#009E73",
        alpha=0.75,
        edgecolor="k",
    )

    ax.axvline(0, color="k", ls="--", lw=1.2)

    pneg = np.mean(d["delta_f_bin_2g_minus_1g_mean"] < 0)

    ax.text(
        0.03,
        0.95,
        rf"Equal-input mocks" "\n"
        rf"$P(\hat{{\Delta}}<0)={pneg:.2f}$" "\n"
        rf"$N={len(d)}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.7"),
    )

    ax.set_xlabel(r"Recovered $\Delta f_{\rm bin}$")
    ax.set_ylabel("Number of mocks")

    fig.tight_layout()
    fig.savefig(outdir / "Fig_realistic_mock_equal_case_hist.pdf")
    fig.savefig(outdir / "Fig_realistic_mock_equal_case_hist.png", dpi=250)
    plt.close(fig)


# ============================================================
# LaTeX table
# ============================================================

def write_latex_table(summary, path):
    """
    Write compact LaTeX table for paper.
    """
    cols = [
        "scenario",
        "N_success",
        "N_failed",
        "true_delta_mean",
        "bias_delta",
        "rmse_delta",
        "coverage95",
        "sign_recovery_rate",
        "P_delta_hat_lt0_equal_case",
        "qc_fail_rate_successful_runs",
    ]

    d = summary[cols].copy()

    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering")
    lines.append(r"\caption{Realistic mock injection--recovery performance.}")
    lines.append(r"\begin{tabular}{lrrrrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"Scenario & $N_{\rm succ}$ & $N_{\rm fail}$ & "
        r"$\Delta_{\rm true}$ & Bias & RMSE & Cov$_{95}$ & "
        r"Sign rec. & $P(\hat{\Delta}<0)$ eq. & QC fail \\"
    )
    lines.append(r"\hline")

    for _, row in d.iterrows():
        scenario = str(row["scenario"]).replace("_", r"\_")
        vals = [
            scenario,
            f"{int(row['N_success'])}",
            f"{int(row['N_failed'])}",
            format_float(row["true_delta_mean"], 3),
            format_float(row["bias_delta"], 3),
            format_float(row["rmse_delta"], 3),
            format_float(row["coverage95"], 2),
            format_float(row["sign_recovery_rate"], 2),
            format_float(row["P_delta_hat_lt0_equal_case"], 2),
            format_float(row["qc_fail_rate_successful_runs"], 2),
        ]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:realistic_mock_recovery}")
    lines.append(r"\end{table*}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default="outputs_realistic_mock/realistic_mock_grid_summary.csv",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: same directory as input file.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if args.output_dir is None:
        outdir = input_path.parent
    else:
        outdir = ensure_dir(args.output_dir)

    df = pd.read_csv(input_path)

    numeric_cols = [
        "true_delta",
        "true_f_bin_1g1g",
        "true_f_bin_2g2g",
        "true_f_bin_mixed",
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
    ]
    df = to_num(df, numeric_cols)

    summary = compute_recovery_table(df)

    summary_csv = outdir / "realistic_mock_recovery_by_scenario.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    latex_path = outdir / "realistic_mock_recovery_table.tex"
    write_latex_table(summary, latex_path)

    plot_delta_recovery(df, outdir)
    plot_bias_by_scenario(summary, outdir)
    plot_equal_case_hist(df, outdir)

    print("Saved:", summary_csv)
    print("Saved:", latex_path)
    print("Saved plots to:", outdir)

    print("\nRecovery summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
