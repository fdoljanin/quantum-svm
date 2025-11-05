import numpy as np
from .types import (
    ExperimentConfig,
    KernelConfig,
    FeatureMapConfig,
    DataConfig,
    SVMConfig,
)

default_shot_based_config = ExperimentConfig(
    name="default_shot_based",
    description="Default shot-based quantum kernel configuration",
    feature_map=FeatureMapConfig(
        type="z",
        feature_dimension=8,
        reps=2,
    ),
    kernel=KernelConfig(
        strategy="shot_based",
        shots=2048,
        optimization_level=1,
        workers=20,
        show_progress=False,
    ),
    data=DataConfig(
        data_path="SUSY.csv.gz",
        nrows=100_000,
        feature_columns=(1, 2, 3, 4, 5, 6, 7, 8),
        target_column=0,
        scale_range=(-np.pi / 4, np.pi / 4),
        scale_factor=1.0,
        train_size=800,
        test_size=1000,
    ),
    svm=SVMConfig(C=1.0),
)

default_statevector_config = ExperimentConfig(
    name="default_statevector",
    description="Default statevector quantum kernel configuration",
    feature_map=FeatureMapConfig(
        type="z",
        feature_dimension=8,
        reps=2,
    ),
    kernel=KernelConfig(
        strategy="statevector",
        cache_statevectors=True,
        show_progress=True,
    ),
    data=DataConfig(
        data_path="SUSY.csv.gz",
        nrows=10_000,
        feature_columns=(1, 2, 3, 4, 5, 6, 7, 8),
        target_column=0,
        scale_range=(-np.pi / 2, np.pi / 2),
        scale_factor=0.5,
        train_size=500,
        test_size=500,
    ),
    svm=SVMConfig(C=1.0),
)

shot_sweep_base = default_shot_based_config
