from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Dict, Any
import numpy as np

@dataclass(frozen=True)
class FeatureMapConfig:
    """Configuration for quantum feature map."""
    type: Literal["z", "zz", "pauli"]
    feature_dimension: int
    reps: int
    paulis: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return {
            'type': self.type,
            'feature_dimension': self.feature_dimension,
            'reps': self.reps,
            'paulis': self.paulis,
        }

@dataclass(frozen=True)
class KernelConfig:
    """Configuration for quantum kernel computation strategy."""
    strategy: Literal["shot_based", "statevector"]

    shots: Optional[int] = 2048
    optimization_level: Optional[int] = 1
    workers: Optional[int] = 20
    show_progress: Optional[bool] = False

    cache_statevectors: Optional[bool] = True

    def to_dict(self) -> dict:
        return {
            'strategy': self.strategy,
            'shots': self.shots,
            'optimization_level': self.optimization_level,
            'workers': self.workers,
            'show_progress': self.show_progress,
            'cache_statevectors': self.cache_statevectors,
        }

@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str = "SUSY.csv.gz"
    nrows: Optional[int] = 100_000
    feature_columns: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    target_column: int = 0

    scale_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4)
    scale_factor: float = 1.0

    train_size: int = 800
    test_size: int = 1000

    def to_dict(self) -> dict:
        return {
            'data_path': self.data_path,
            'nrows': self.nrows,
            'feature_columns': list(self.feature_columns),
            'target_column': self.target_column,
            'scale_range': list(self.scale_range),
            'scale_factor': self.scale_factor,
            'train_size': self.train_size,
            'test_size': self.test_size,
        }

@dataclass(frozen=True)
class SVMConfig:
    """Configuration for SVM training."""
    C: float = 1.0
    kernel: str = "precomputed"

    def to_dict(self) -> dict:
        return {
            'C': self.C,
            'kernel': self.kernel,
        }

@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration."""
    feature_map: FeatureMapConfig
    kernel: KernelConfig
    data: DataConfig
    svm: SVMConfig = field(default_factory=lambda: SVMConfig())

    name: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'feature_map': self.feature_map.to_dict(),
            'kernel': self.kernel.to_dict(),
            'data': self.data.to_dict(),
            'svm': self.svm.to_dict(),
        }

@dataclass
class MetricsResult:
    """Results from experiment evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def to_dict(self) -> dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
        }

@dataclass
class ExperimentResult:
    """Complete experiment results."""
    config: ExperimentConfig
    metrics: MetricsResult
    predictions: np.ndarray
    kernel_train: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (excludes predictions and kernel data)."""
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
        }
