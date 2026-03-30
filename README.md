# Binary Population in Galactic Globular Clusters: A Bayesian Census of Multiple Populations

This repository contains the analysis code and results for the comprehensive study of binary star fractions in first-generation (1G) and second-generation (2G) stars across 56 Galactic globular clusters. Using a hierarchical Bayesian mixture model applied to HUGS survey data, we uncover a universal depletion of 2G binaries and its dynamical dependence on core collision rates.

## Repository Contents

- **`cal_fbin_ratio_bayesian_v2.py`** – Main Python script implementing the five‑component hierarchical Bayesian model to infer binary fractions and the ratio \( R = f_{\mathrm{bin}}^{\mathrm{2G}} / f_{\mathrm{bin}}^{\mathrm{1G}} \).
- **`GC_Binary_Results_Summary.csv`** – Summary table of inferred binary ratios, uncertainties, and ancillary parameters (escape velocity, collision rate, flags) for all 56 clusters.
- **`HBM_model_fitted_well.zip`** – Archive containing diagnostic plots and posterior distributions for the fitted Bayesian model.
- **`Surface Density Profile fitting Plots.zip`** – Archive with surface density profile fits used to derive cluster structural parameters.

## Key Results

- **Universal 2G binary exhaustion:** The global stacked posterior peaks at \( R \approx 0.01 \), with \( P(R<1) > 99.9\% \).
- **Heggie‑Hut signature:** A strong positive correlation (\( \rho = 0.73 \)) between \( R \) and the core collision rate \( \Gamma \) reveals that dynamical evolution preferentially destroys wide 1G binaries in dense cores.
- **Formation constraint:** No correlation with global escape velocity indicates that the microscopic density of the 2G‑forming gas reservoir, not the cluster’s global potential, governs binary survival.

## Requirements

The analysis requires Python 3.8+ with the following packages:

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `pymc` (or `pymc3`) for Bayesian inference
- `arviz` for posterior diagnostics

Install dependencies via:

```bash
pip install numpy scipy pandas matplotlib pymc arviz
