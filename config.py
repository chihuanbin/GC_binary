# gc_binary_pipeline/config.py

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class ClusterConfig:
    cluster_name: str
    file_path: str
    

    # MS magnitude range in F606W
    mag_min: float
    mag_max: float

    # sample size for MCMC acceleration
    sample_size: int = 8000
    random_seed: int = 42

    # quality cuts
    membership_min: float = 90.0
    q_min: float = 0.90
    rms_max: float = 0.20
    sharp_abs_max: float = 0.30
    require_membership: bool = True

    # differential reddening
    apply_dr: bool = True
    dr_n_neighbors: int = 60
    apply_differential_reddening=True
    # chromosome map fiducials
    fiducial_bins: int = 25
    blue_percentile: float = 5.0
    red_percentile: float = 95.0
    min_bin_count: int = 40
    fiducial_smoothing: float = 0.0005
    # Robust clipping in chromosome-map-like coordinates
    delta_abs_max: float = 3.0
    template_delta_abs_max: float = 2.0

    # Avoid synthetic binaries becoming brighter than fiducial range
    binary_primary_mag_buffer: float = 0.85

    # photometric uncertainty floor in normalized chromosome-map coordinates
    delta_error_floor: float = 0.01

    # initial population split
    init_gmm_components: int = 2
    init_prob_threshold: float = 0.80

    # binary forward model
    q_min_binary: float = 0.5
    q_max_binary: float = 1.0
    n_binary_template: int = 30000
    mass_luminosity_alpha: float = 4.0

    # template GMMs
    gmm_components_single: int = 4
    gmm_components_binary: int = 5
    gmm_components_field: int = 3
    gmm_reg_covar: float = 1e-4

    # PyMC sampling
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.95

    reverse_population_assignment: bool = False

    # output
    output_dir: str = "outputs"


# HUGS catalog column mapping.
# Input file has no header. Python is 0-indexed.
# Based on user-provided format:
# Col. 1,2 -> X,Y
# Col. 3 -> F275W, Col. 4 -> rms, Col. 5 -> Q, Col. 6 -> sharp, ...
HUGS_COLMAP: Dict[int, str] = {
    0: "X",
    1: "Y",

    2: "F275W",
    3: "E275",
    4: "Q275",
    5: "S275",
    6: "N275_found",
    7: "N275_good",

    8: "F336W",
    9: "E336",
    10: "Q336",
    11: "S336",
    12: "N336_found",
    13: "N336_good",

    14: "F435W",
    15: "E435",
    16: "Q435",
    17: "S435",
    18: "N435_found",
    19: "N435_good",

    20: "F606W",
    21: "E606",
    22: "Q606",
    23: "S606",
    24: "N606_found",
    25: "N606_good",

    26: "F814W",
    27: "E814",
    28: "Q814",
    29: "S814",
    30: "N814_found",
    31: "N814_good",

    32: "Prob",
    33: "RA",
    34: "Dec",
    35: "ID",
    36: "Iter",
}


# Approximate extinction coefficients.
# These are placeholders and should be replaced by cluster/filter-specific values
# if available.
EXTINCTION_COEFF = {
    "F275W": 6.10,
    "F336W": 5.10,
    "F435W": 4.15,
    "F606W": 2.90,
    "F814W": 1.85,
}
