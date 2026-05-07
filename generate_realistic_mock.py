#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_realistic_mock.py

Generate Milone-like realistic mock HST catalogues for binary-fraction
injection--recovery tests.

The mock is intentionally more complex than the inference model:
  - curved/non-Gaussian 1G morphology
  - multiple 2G subpopulations
  - 1G--1G, 2G--2G, and mixed binaries
  - magnitude-dependent, correlated photometric errors
  - spatially correlated differential reddening
  - non-uniform field contamination
  - optional crowding/outlier tail

Output:
  mock_catalog.txt          whitespace table, HST-like columns
  mock_truth_per_star.csv   per-star truth table
  mock_metadata.json        injected global truth
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
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

# from generate_realistic_mock import generate_realistic_mock
from gc_binary_pipeline.config import ClusterConfig
from gc_binary_pipeline.run_cluster import run_cluster




# ============================================================
# Utilities
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def flux_from_mag(mag):
    return 10.0 ** (-0.4 * mag)


def mag_from_flux(flux):
    flux = np.maximum(flux, 1e-300)
    return -2.5 * np.log10(flux)


def add_fluxes(mag1, mag2):
    return mag_from_flux(flux_from_mag(mag1) + flux_from_mag(mag2))


def robust_clip01(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


# ============================================================
# Spatial model
# ============================================================

def sample_cluster_positions(n, rng, r_scale=1.0, r_max=5.0):
    """
    Simple projected Plummer-like radial distribution.
    """
    u = rng.uniform(size=n)
    r = r_scale * np.sqrt(u / np.maximum(1.0 - u, 1e-6))
    r = np.minimum(r, r_max)

    theta = rng.uniform(0, 2 * np.pi, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, r


def sample_field_positions(n, rng, r_max=5.0):
    """
    Uniform field positions in square footprint.
    """
    x = rng.uniform(-r_max, r_max, size=n)
    y = rng.uniform(-r_max, r_max, size=n)
    r = np.sqrt(x**2 + y**2)
    return x, y, r


# ============================================================
# Magnitude and population model
# ============================================================

def sample_luminosity_function(n, rng, mag_min, mag_max, slope=0.55):
    """
    Simple increasing luminosity function toward faint magnitudes.
    """
    u = rng.uniform(size=n)
    a = slope
    if abs(a) < 1e-8:
        return rng.uniform(mag_min, mag_max, size=n)

    e0 = np.exp(a * mag_min)
    e1 = np.exp(a * mag_max)
    mag = np.log(e0 + u * (e1 - e0)) / a
    return mag


def sample_population_labels(r, rng, f_2g_global=0.70, radial_gradient=True):
    """
    Population labels for cluster stars.
    1G = 1, 2G = 2.

    If radial_gradient=True, 2G is more centrally concentrated.
    """
    n = len(r)

    if radial_gradient:
        # 2G fraction high in center, lower outside.
        # Normalized to be roughly around f_2g_global.
        p2 = f_2g_global + 0.18 * np.exp(-(r / 1.4) ** 2) - 0.10 * sigmoid((r - 2.8) / 0.7)
        p2 = robust_clip01(p2, 0.25, 0.95)
    else:
        p2 = np.full(n, f_2g_global)

    pop = np.where(rng.uniform(size=n) < p2, 2, 1)
    return pop, p2


def sample_2g_subpopulation(n, rng, weights=(0.45, 0.35, 0.20)):
    """
    2G subpopulation labels: 0, 1, 2.
    """
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return rng.choice(len(weights), size=n, p=weights)


# ============================================================
# Milone-like morphology in pseudo-colour space
# ============================================================

def intrinsic_chromosome_coordinates(m606, pop, sub2g, rng):
    """
    Generate intrinsic chromosome-map-like coordinates:
      d1 ~ UV pseudo-colour residual
      d2 ~ optical/UV residual

    The morphology is deliberately non-Gaussian and curved.
    """
    n = len(m606)
    x = m606 - np.median(m606)

    d1 = np.zeros(n)
    d2 = np.zeros(n)

    # 1G: curved, mildly broadened, slightly skewed
    idx1 = pop == 1
    n1 = idx1.sum()
    eps1 = rng.normal(size=n1)
    eps2 = rng.normal(size=n1)
    skew = rng.gamma(shape=2.0, scale=0.010, size=n1) - 0.020

    d1[idx1] = (
        -0.020
        + 0.010 * x[idx1]
        + 0.018 * x[idx1] ** 2
        + 0.030 * eps1
        + skew
    )

    d2[idx1] = (
        0.000
        - 0.006 * x[idx1]
        + 0.010 * x[idx1] ** 2
        + 0.026 * (0.45 * eps1 + np.sqrt(1 - 0.45**2) * eps2)
    )

    # 2G: mixture of curved sub-sequences
    idx2 = pop == 2
    for k in [0, 1, 2]:
        idx = idx2 & (sub2g == k)
        nk = idx.sum()
        if nk == 0:
            continue

        eps1 = rng.normal(size=nk)
        eps2 = rng.normal(size=nk)

        # Milone-like extended/multiple 2G structure
        offset_d1 = [0.16, 0.32, 0.52][k]
        offset_d2 = [0.08, 0.18, 0.31][k]
        curve1 = [0.014, -0.010, 0.022][k]
        curve2 = [-0.006, 0.015, -0.018][k]
        width1 = [0.030, 0.040, 0.052][k]
        width2 = [0.024, 0.034, 0.046][k]
        rho = [0.2, -0.35, 0.55][k]

        d1[idx] = (
            offset_d1
            + 0.010 * x[idx]
            + curve1 * x[idx] ** 2
            + width1 * eps1
        )

        d2[idx] = (
            offset_d2
            - 0.005 * x[idx]
            + curve2 * x[idx] ** 2
            + width2 * (rho * eps1 + np.sqrt(1 - rho**2) * eps2)
        )

    return d1, d2


# ============================================================
# Convert pseudo morphology to HST-like magnitudes
# ============================================================

def chromosome_to_filters(m606, d1, d2):
    """
    Create HST-like magnitudes F275W, F336W, F438W, F606W, F814W.

    This is not meant to be a physical stellar model. It is designed to
    produce realistic-looking multi-filter morphology with correlated
    chromosome-map structure.

    Definitions used internally:
      c275336 = F275W - F336W
      c336438 = F336W - F438W
      c606814 = F606W - F814W
    """
    x = m606 - np.median(m606)

    # Smooth single-star colour-magnitude trends
    c606814_base = 0.70 + 0.10 * x + 0.025 * x**2
    c336438_base = 0.52 + 0.08 * x + 0.015 * x**2
    c275336_base = 1.35 + 0.16 * x + 0.035 * x**2

    # Inject chromosome offsets
    c275336 = c275336_base + 0.95 * d1 + 0.20 * d2
    c336438 = c336438_base - 0.30 * d1 + 0.85 * d2
    c606814 = c606814_base + 0.10 * d1 - 0.05 * d2

    F606W = m606
    F814W = F606W - c606814
    F438W = F606W + 0.85 + 0.10 * x
    F336W = F438W + c336438
    F275W = F336W + c275336

    return F275W, F336W, F438W, F606W, F814W


# ============================================================
# Binary model
# ============================================================

def sample_binary_types(pop, rng, f_bin_1g, f_bin_2g, f_bin_mixed):
    """
    Assign binary types:
      0 = single
      1 = 1G1G binary
      2 = 2G2G binary
      3 = mixed 1G2G binary

    For mock truth:
      - 1G primaries may become 1G1G or mixed
      - 2G primaries may become 2G2G or mixed
    """
    n = len(pop)
    btype = np.zeros(n, dtype=int)

    idx1 = pop == 1
    idx2 = pop == 2

    u = rng.uniform(size=n)

    # For 1G primaries
    p11 = f_bin_1g
    pmix1 = f_bin_mixed
    btype[idx1 & (u < p11)] = 1
    btype[idx1 & (u >= p11) & (u < p11 + pmix1)] = 3

    # For 2G primaries
    p22 = f_bin_2g
    pmix2 = f_bin_mixed
    btype[idx2 & (u < p22)] = 2
    btype[idx2 & (u >= p22) & (u < p22 + pmix2)] = 3

    return btype


def sample_mass_ratio(n, rng, q_min=0.45, mode="flat"):
    """
    Mass-ratio distribution.
    """
    u = rng.uniform(size=n)

    if mode == "flat":
        q = q_min + (1 - q_min) * u
    elif mode == "rising":
        q = q_min + (1 - q_min) * np.sqrt(u)
    elif mode == "falling":
        q = q_min + (1 - q_min) * (1 - np.sqrt(1 - u))
    else:
        raise ValueError(f"Unknown q mode: {mode}")

    return q


def secondary_magnitudes_from_primary(
    F275W, F336W, F438W, F606W, F814W, q, rng
):
    """
    Approximate secondary magnitudes using filter-dependent mass-luminosity
    exponents. A small random mismatch is added to avoid identical forward
    and inference templates.
    """
    # Alpha controls L ~ M^alpha. Slight filter dependence.
    alpha_275 = 4.8
    alpha_336 = 4.5
    alpha_438 = 4.2
    alpha_606 = 3.8
    alpha_814 = 3.5

    # m2 = m1 - 2.5 log10(q^alpha) = m1 - 2.5 alpha log10(q)
    # Since q<1, log10(q)<0, so m2>m1.
    jitter = rng.normal(0.0, 0.015, size=len(q))

    F275W_2 = F275W - 2.5 * alpha_275 * np.log10(q) + jitter
    F336W_2 = F336W - 2.5 * alpha_336 * np.log10(q) + jitter
    F438W_2 = F438W - 2.5 * alpha_438 * np.log10(q) + jitter
    F606W_2 = F606W - 2.5 * alpha_606 * np.log10(q) + jitter
    F814W_2 = F814W - 2.5 * alpha_814 * np.log10(q) + jitter

    return F275W_2, F336W_2, F438W_2, F606W_2, F814W_2


def apply_unresolved_binaries(
    F275W,
    F336W,
    F438W,
    F606W,
    F814W,
    btype,
    rng,
    q_mode="flat",
):
    """
    Add unresolved binary flux to stars with btype > 0.
    """
    n = len(F606W)
    q = np.full(n, np.nan)

    idx = btype > 0
    nb = idx.sum()
    if nb == 0:
        return F275W, F336W, F438W, F606W, F814W, q

    q[idx] = sample_mass_ratio(nb, rng, q_min=0.45, mode=q_mode)

    sec = secondary_magnitudes_from_primary(
        F275W[idx],
        F336W[idx],
        F438W[idx],
        F606W[idx],
        F814W[idx],
        q[idx],
        rng,
    )

    F275W_new = F275W.copy()
    F336W_new = F336W.copy()
    F438W_new = F438W.copy()
    F606W_new = F606W.copy()
    F814W_new = F814W.copy()

    F275W_new[idx] = add_fluxes(F275W[idx], sec[0])
    F336W_new[idx] = add_fluxes(F336W[idx], sec[1])
    F438W_new[idx] = add_fluxes(F438W[idx], sec[2])
    F606W_new[idx] = add_fluxes(F606W[idx], sec[3])
    F814W_new[idx] = add_fluxes(F814W[idx], sec[4])

    return F275W_new, F336W_new, F438W_new, F606W_new, F814W_new, q


# ============================================================
# Differential reddening and photometric errors
# ============================================================

def make_dr_field(x, y, rng, amp=0.030, n_blobs=12, scale_range=(0.8, 2.5)):
    """
    Spatially correlated differential reddening field.
    """
    e = np.zeros_like(x, dtype=float)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    for _ in range(n_blobs):
        xc = rng.uniform(xmin, xmax)
        yc = rng.uniform(ymin, ymax)
        scale = rng.uniform(scale_range[0], scale_range[1])
        a = rng.normal(0.0, amp)
        e += a * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * scale**2))

    e -= np.median(e)
    return e


def apply_differential_reddening(
    F275W,
    F336W,
    F438W,
    F606W,
    F814W,
    x,
    y,
    rng,
    amp=0.030,
):
    """
    Apply DR with approximate filter coefficients.
    """
    e = make_dr_field(x, y, rng, amp=amp)

    # Approximate relative extinction coefficients.
    A275 = 6.1 * e
    A336 = 5.1 * e
    A438 = 4.1 * e
    A606 = 2.9 * e
    A814 = 1.8 * e

    return (
        F275W + A275,
        F336W + A336,
        F438W + A438,
        F606W + A606,
        F814W + A814,
        e,
    )


def phot_error_sigma(mag, floor=0.004, scale=0.010, m0=19.5, tau=1.25):
    """
    Artificial-star-like magnitude-dependent error.
    """
    return floor + scale * np.exp((mag - m0) / tau)


def apply_photometric_errors(
    F275W,
    F336W,
    F438W,
    F606W,
    F814W,
    rng,
    error_scale=1.0,
):
    """
    Apply correlated photometric errors.
    """
    mags = [F275W, F336W, F438W, F606W, F814W]
    out = []

    # Per-filter error scale. UV is noisier.
    factors = [1.8, 1.5, 1.2, 0.9, 0.9]

    # Shared common-mode error creates colour correlations.
    common = rng.normal(0.0, 1.0, size=len(F606W))

    err_cols = []

    for mag, fac in zip(mags, factors):
        sig = error_scale * fac * phot_error_sigma(F606W)
        independent = rng.normal(0.0, sig)
        correlated = 0.35 * sig * common
        err = independent + correlated
        out.append(mag + err)
        err_cols.append(sig)

    return (*out, *err_cols)


# ============================================================
# Field contamination
# ============================================================

def generate_field_stars(n_field, rng, mag_min, mag_max, r_max=5.0):
    """
    Generate non-uniform CMD field contamination.
    """
    x, y, r = sample_field_positions(n_field, rng, r_max=r_max)

    # Non-uniform field: preferential gradient along x.
    # Rejection sample to get spatial non-uniformity.
    keep = []
    attempts = 0
    while len(keep) < n_field and attempts < 50:
        xx, yy, rr = sample_field_positions(n_field, rng, r_max=r_max)
        prob = robust_clip01(0.55 + 0.35 * xx / r_max + 0.15 * np.sin(yy), 0.05, 1.0)
        u = rng.uniform(size=n_field)
        idx = np.where(u < prob)[0]
        keep.extend(list(zip(xx[idx], yy[idx], rr[idx])))
        attempts += 1

    keep = keep[:n_field]
    x = np.array([v[0] for v in keep])
    y = np.array([v[1] for v in keep])
    r = np.array([v[2] for v in keep])

    m606 = sample_luminosity_function(n_field, rng, mag_min, mag_max, slope=0.35)

    # Broad field colour distribution, partially overlapping cluster.
    xmag = m606 - np.median(m606)
    c606814 = 0.75 + 0.18 * xmag + rng.normal(0, 0.18, size=n_field)
    c336438 = 0.65 + 0.12 * xmag + rng.normal(0, 0.32, size=n_field)
    c275336 = 1.45 + 0.18 * xmag + rng.normal(0, 0.45, size=n_field)

    # Add a red contaminant tail.
    tail = rng.uniform(size=n_field) < 0.20
    c606814[tail] += rng.normal(0.35, 0.12, size=tail.sum())
    c336438[tail] += rng.normal(0.25, 0.15, size=tail.sum())

    F606W = m606
    F814W = F606W - c606814
    F438W = F606W + 0.85 + 0.10 * xmag
    F336W = F438W + c336438
    F275W = F336W + c275336

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "r": r,
            "F275W": F275W,
            "F336W": F336W,
            "F438W": F438W,
            "F606W": F606W,
            "F814W": F814W,
            "pop_truth": 0,
            "subpop_truth": -1,
            "binary_type_truth": -1,
            "q_truth": np.nan,
            "is_field_truth": True,
            "dr_truth": 0.0,
        }
    )


# ============================================================
# Main generator
# ============================================================

def generate_realistic_mock(
    output_dir,
    mock_name="mock",
    seed=123,
    n_total=5000,
    mag_min=18.0,
    mag_max=21.5,
    f_2g=0.70,
    f_bin_1g=0.06,
    f_bin_2g=0.03,
    f_bin_mixed=0.01,
    w_field=0.03,
    n_2g_subpops=3,
    q_mode="flat",
    dr_amp=0.030,
    phot_error_scale=1.0,
    crowding_outlier_frac=0.01,
    radial_gradient=True,
):
    """
    Generate one realistic mock catalogue.
    """
    rng = np.random.default_rng(seed)
    output_dir = ensure_dir(output_dir)

    n_field = int(np.round(n_total * w_field))
    n_cluster = n_total - n_field

    # Cluster positions
    x, y, r = sample_cluster_positions(n_cluster, rng, r_scale=1.0, r_max=5.0)

    # Magnitudes
    m606 = sample_luminosity_function(n_cluster, rng, mag_min, mag_max, slope=0.55)

    # Population labels
    pop, p2_local = sample_population_labels(
        r, rng, f_2g_global=f_2g, radial_gradient=radial_gradient
    )

    sub2g = np.full(n_cluster, -1, dtype=int)
    idx2 = pop == 2
    sub2g[idx2] = sample_2g_subpopulation(idx2.sum(), rng)

    # Intrinsic morphology
    d1, d2 = intrinsic_chromosome_coordinates(m606, pop, sub2g, rng)

    # Convert to filters
    F275W, F336W, F438W, F606W, F814W = chromosome_to_filters(m606, d1, d2)

    # Binary types
    btype = sample_binary_types(
        pop,
        rng,
        f_bin_1g=f_bin_1g,
        f_bin_2g=f_bin_2g,
        f_bin_mixed=f_bin_mixed,
    )

    # Apply binary flux
    F275W, F336W, F438W, F606W, F814W, q = apply_unresolved_binaries(
        F275W,
        F336W,
        F438W,
        F606W,
        F814W,
        btype,
        rng,
        q_mode=q_mode,
    )

    # Differential reddening
    if dr_amp > 0:
        F275W, F336W, F438W, F606W, F814W, dr = apply_differential_reddening(
            F275W,
            F336W,
            F438W,
            F606W,
            F814W,
            x,
            y,
            rng,
            amp=dr_amp,
        )
    else:
        dr = np.zeros(n_cluster)

    # Photometric errors
    (
        F275W,
        F336W,
        F438W,
        F606W,
        F814W,
        e275,
        e336,
        e438,
        e606,
        e814,
    ) = apply_photometric_errors(
        F275W,
        F336W,
        F438W,
        F606W,
        F814W,
        rng,
        error_scale=phot_error_scale,
    )

    # Crowding/outlier tail
    is_crowding_outlier = rng.uniform(size=n_cluster) < crowding_outlier_frac
    F275W[is_crowding_outlier] += rng.normal(0.0, 0.25, size=is_crowding_outlier.sum())
    F336W[is_crowding_outlier] += rng.normal(0.0, 0.20, size=is_crowding_outlier.sum())
    F438W[is_crowding_outlier] += rng.normal(0.0, 0.15, size=is_crowding_outlier.sum())
    F606W[is_crowding_outlier] += rng.normal(0.0, 0.10, size=is_crowding_outlier.sum())
    F814W[is_crowding_outlier] += rng.normal(0.0, 0.10, size=is_crowding_outlier.sum())

    cluster_df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "r": r,
            "F275W": F275W,
            "F336W": F336W,
            "F438W": F438W,
            "F606W": F606W,
            "F814W": F814W,
            "e_F275W": e275,
            "e_F336W": e336,
            "e_F438W": e438,
            "e_F606W": e606,
            "e_F814W": e814,
            "pop_truth": pop,
            "subpop_truth": sub2g,
            "binary_type_truth": btype,
            "q_truth": q,
            "is_field_truth": False,
            "dr_truth": dr,
            "is_crowding_outlier_truth": is_crowding_outlier,
        }
    )

    # Field stars
    if n_field > 0:
        field_df = generate_field_stars(n_field, rng, mag_min, mag_max, r_max=5.0)

        # Add missing columns
        for c in cluster_df.columns:
            if c not in field_df.columns:
                if c.startswith("e_"):
                    field_df[c] = np.nan
                elif c == "is_crowding_outlier_truth":
                    field_df[c] = False
                else:
                    field_df[c] = np.nan

        field_df = field_df[cluster_df.columns]
        df = pd.concat([cluster_df, field_df], ignore_index=True)
    else:
        df = cluster_df.copy()

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Add common aliases that a HUGS-like loader may expect
    df["m_F275W"] = df["F275W"]
    df["m_F336W"] = df["F336W"]
    df["m_F438W"] = df["F438W"]
    df["m_F606W"] = df["F606W"]
    df["m_F814W"] = df["F814W"]

    # Compute realised truth among cluster members only
    cl = df[~df["is_field_truth"].astype(bool)].copy()
    n1 = np.sum(cl["pop_truth"] == 1)
    n2 = np.sum(cl["pop_truth"] == 2)

    realised_f_1g1g = np.sum((cl["pop_truth"] == 1) & (cl["binary_type_truth"] == 1)) / max(n1, 1)
    realised_f_2g2g = np.sum((cl["pop_truth"] == 2) & (cl["binary_type_truth"] == 2)) / max(n2, 1)

    # Mixed defined per all cluster stars here
    realised_f_mixed = np.sum(cl["binary_type_truth"] == 3) / max(len(cl), 1)

    metadata = {
        "mock_name": mock_name,
        "seed": int(seed),
        "n_total": int(n_total),
        "n_cluster": int(n_cluster),
        "n_field": int(n_field),
        "mag_min": float(mag_min),
        "mag_max": float(mag_max),
        "input_f_2g": float(f_2g),
        "input_f_bin_1g1g": float(f_bin_1g),
        "input_f_bin_2g2g": float(f_bin_2g),
        "input_f_bin_mixed": float(f_bin_mixed),
        "input_delta": float(f_bin_2g - f_bin_1g),
        "input_w_field": float(w_field),
        "q_mode": q_mode,
        "dr_amp": float(dr_amp),
        "phot_error_scale": float(phot_error_scale),
        "crowding_outlier_frac": float(crowding_outlier_frac),
        "radial_gradient": bool(radial_gradient),
        "realised_f_bin_1g1g": float(realised_f_1g1g),
        "realised_f_bin_2g2g": float(realised_f_2g2g),
        "realised_f_bin_mixed": float(realised_f_mixed),
        "realised_delta": float(realised_f_2g2g - realised_f_1g1g),
        "realised_w_field": float(n_field / n_total),
        "realised_w_2g_cluster": float(n2 / max(n_cluster, 1)),
    }

    catalog_path = output_dir / f"{mock_name}_catalog.txt"
    truth_path = output_dir / f"{mock_name}_truth_per_star.csv"
    metadata_path = output_dir / f"{mock_name}_metadata.json"

    # Save catalogue as whitespace-separated HST-like table
    # The truth columns are kept in the catalogue too, but your production
    # loader can ignore them.
    df.to_csv(catalog_path, sep=" ", index=False, float_format="%.8f")
    df.to_csv(truth_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return catalog_path, truth_path, metadata_path, metadata


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--output-dir", required=True)
    p.add_argument("--mock-name", default="mock")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--n-total", type=int, default=5000)
    p.add_argument("--mag-min", type=float, default=18.0)
    p.add_argument("--mag-max", type=float, default=21.5)

    p.add_argument("--f-2g", type=float, default=0.70)
    p.add_argument("--f-bin-1g", type=float, default=0.06)
    p.add_argument("--f-bin-2g", type=float, default=0.03)
    p.add_argument("--f-bin-mixed", type=float, default=0.01)
    p.add_argument("--w-field", type=float, default=0.03)

    p.add_argument("--q-mode", default="flat", choices=["flat", "rising", "falling"])
    p.add_argument("--dr-amp", type=float, default=0.030)
    p.add_argument("--phot-error-scale", type=float, default=1.0)
    p.add_argument("--crowding-outlier-frac", type=float, default=0.01)

    p.add_argument("--no-radial-gradient", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    catalog_path, truth_path, metadata_path, metadata = generate_realistic_mock(
        output_dir=args.output_dir,
        mock_name=args.mock_name,
        seed=args.seed,
        n_total=args.n_total,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        f_2g=args.f_2g,
        f_bin_1g=args.f_bin_1g,
        f_bin_2g=args.f_bin_2g,
        f_bin_mixed=args.f_bin_mixed,
        w_field=args.w_field,
        q_mode=args.q_mode,
        dr_amp=args.dr_amp,
        phot_error_scale=args.phot_error_scale,
        crowding_outlier_frac=args.crowding_outlier_frac,
        radial_gradient=not args.no_radial_gradient,
    )

    print("Saved catalogue:", catalog_path)
    print("Saved truth:", truth_path)
    print("Saved metadata:", metadata_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
