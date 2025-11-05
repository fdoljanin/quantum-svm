"""
Configuration for shot count sweep experiment.

Reproduces the original shots.py experiment with varying shot counts.
"""

import numpy as np
from qsvm.config import (
    ExperimentConfig,
    KernelConfig,
    FeatureMapConfig,
    DataConfig,
    SVMConfig,
)

base_config = ExperimentConfig(
    name="shot_sweep",
    description="Shot count sweep: 1 to 4096 shots",
    feature_map=FeatureMapConfig(
        type="z",
        feature_dimension=8,
        reps=2,
    ),
    kernel=KernelConfig(
        strategy="shot_based",
        shots=2048,
        optimization_level=1,
        workers=72,
        show_progress=False,
    ),
    data=DataConfig(
        data_path="SUSY.csv.gz",
        nrows=100_000,
        train_size=800,
        test_size=1000,
        scale_range=(-np.pi / 4, np.pi / 4),
    ),
    svm=SVMConfig(C=1.0),
)

config = base_config
output_filename = "shot_sweep_2048.json"

shot_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

def get_configs():
    """Generate configs for all shot counts (for batch runner)."""
    from dataclasses import replace

    configs = []
    for shots in shot_counts:
        kernel_config = replace(base_config.kernel, shots=shots)
        exp_config = replace(
            base_config,
            name=f"shot_sweep_{shots}",
            kernel=kernel_config,
        )
        configs.append(exp_config)

    return configs
