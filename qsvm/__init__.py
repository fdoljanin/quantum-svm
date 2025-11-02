"""
Quantum Support Vector Machine (QSVM) library.

Modular, extensible implementation of quantum kernel methods for SVM classification.
Designed for high-performance experimentation on cluster environments and genetic
algorithm optimization.
"""

__version__ = "2.0.0"

from .models import QSVM
from .kernels import QuantumKernel, ShotBasedKernel, StatevectorKernel
from .config import (
    ExperimentConfig,
    KernelConfig,
    FeatureMapConfig,
    DataConfig,
    SVMConfig,
    MetricsResult,
    ExperimentResult,
    default_shot_based_config,
    default_statevector_config,
)

__all__ = [
    "QSVM",
    "QuantumKernel",
    "ShotBasedKernel",
    "StatevectorKernel",
    "ExperimentConfig",
    "KernelConfig",
    "FeatureMapConfig",
    "DataConfig",
    "SVMConfig",
    "MetricsResult",
    "ExperimentResult",
    "default_shot_based_config",
    "default_statevector_config",
]
