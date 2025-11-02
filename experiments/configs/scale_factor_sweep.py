"""
Configuration for scale factor sweep experiment.

Tests how the scaling factor affects Z feature map performance,
ranging from 2**0 (1.0) to 2**-10 (approximately 0.00098).
"""

import numpy as np
from qsvm.config import (
    ExperimentConfig,
    KernelConfig,
    FeatureMapConfig,
    DataConfig,
    SVMConfig,
)

# Base configuration
base_config = ExperimentConfig(
    name="scale_factor_sweep",
    description="Scale factor sweep: 2**0 to 2**-10 for Z feature map",
    feature_map=FeatureMapConfig(
        type="z",
        feature_dimension=8,
        reps=2,
    ),
    kernel=KernelConfig(
        strategy="statevector",
        workers=6,  # Cluster configuration
    ),
    data=DataConfig(
        data_path="raw/SUSY.csv.gz",
        nrows=1000,
        train_size=500,
        test_size=500,
        scale_range=(-np.pi / 4, np.pi / 4),
        scale_factor=1.0,  # Will be overridden
    ),
    svm=SVMConfig(C=1.0),
)

# For single experiment, use base config
config = base_config
output_filename = "scale_factor_sweep_1.0.json"

# For batch experiments (used by batch runner):
# Generate scale factors from 2**0 to 2**-10
scale_factor_exponents = list(range(0, -3, -1))  # [0, -1, -2, ..., -10]
scale_factors = [2**exp for exp in scale_factor_exponents]

def get_configs():
    """Generate configs for all scale factors (for batch runner)."""
    from dataclasses import replace

    configs = []
    for sf in scale_factors:
        # Create new config with updated scale_factor
        data_config = replace(base_config.data, scale_factor=sf)
        exp_config = replace(
            base_config,
            name=f"scale_factor_sweep_{sf:.6f}",
            data=data_config,
        )
        configs.append(exp_config)

    return configs
