# Quantum SVM Refactoring for Modularity and Extensibility

## Overview

Refactor the quantum SVM codebase to enable modular, extensible experimentation with focus on genetic algorithm integration and 72-core cluster execution. Transform the current flat structure with duplicate implementations into a clean, strategy-pattern-based architecture where all components (kernels, feature maps, data pipelines, configurations) are easily swappable and composable.

## Current State Analysis

### Existing Structure
- **shots.py**: Shot-based quantum kernel with parallel execution (ProcessPoolExecutor)
- **svm_susy_quantum_eval_0.ipynb**: Statevector-based kernel with caching
- **svm_susy_quantum_eval_1.ipynb**: Hyperparameter experiments
- **SUSY.csv.gz**: Particle physics dataset (922MB)

### Key Problems
1. **Duplicate implementations**: Two QCalculator classes with different approaches
2. **Hard-coded configuration**: Parameters scattered throughout code
3. **No reusability**: Cannot easily compose experiments programmatically
4. **Limited extensibility**: Adding new kernels, feature maps, or datasets requires significant refactoring
5. **Poor GA integration**: Cannot easily generate and test configurations

### Key Discoveries
- Both implementations share identical workflow: `calculate_kernel()` → `svm()` → `predict()`
- Shot-based uses parallel processing (ProcessPoolExecutor), statevector uses caching
- Experiments vary: feature maps (Z/ZZ/Pauli), shots (1-4096), training sizes (100-25600), C values (0.5-5.0)
- Data pipeline consistent: Load → Clean → Scale → Split
- All experiments use same 8 features, same metrics (accuracy, precision, recall, f1)

## Desired End State

### Architecture
```
qsvm/
├── __init__.py                      # Main exports
├── kernels/
│   ├── __init__.py
│   ├── base.py                      # QuantumKernel ABC (Strategy interface)
│   ├── shot_based.py                # Parallel shot-based implementation
│   └── statevector.py               # Cached statevector implementation
├── feature_maps/
│   ├── __init__.py
│   └── factory.py                   # Feature map creation from config
├── data/
│   ├── __init__.py
│   └── pipeline.py                  # DataPipeline class
├── models/
│   ├── __init__.py
│   └── qsvm.py                      # QSVM orchestrator (Strategy context)
├── config/
│   ├── __init__.py
│   ├── types.py                     # Config dataclasses
│   └── defaults.py                  # Default configurations
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                   # MetricsTracker, JSON results
└── utils/
    ├── __init__.py
    └── parallel.py                  # Parallel execution helpers

experiments/
├── configs/
│   ├── __init__.py
│   ├── shot_sweep.py                # Configuration for shot count experiment
│   ├── feature_map_comparison.py    # Configuration for feature map experiments
│   └── hyperparameter_search.py     # Configuration for C/reps experiments
└── run_experiment.py                # Generic experiment runner

notebooks/
├── 01_interactive_demo.ipynb        # Import qsvm module and run experiments
└── 02_results_analysis.ipynb        # Analyze JSON results
```

### Usage After Refactoring

**For Genetic Algorithms:**
```python
from qsvm.config.types import ExperimentConfig, KernelConfig, FeatureMapConfig
from qsvm.models import QSVM

# GA generates configuration
config = ExperimentConfig(
    kernel=KernelConfig(strategy="shot_based", shots=2048, workers=72),
    feature_map=FeatureMapConfig(type="z", reps=2),
    data=DataConfig(train_size=800, test_size=1000, scale_range=(-np.pi/4, np.pi/4)),
    svm_C=1.0,
)

# Run experiment
qsvm = QSVM.from_config(config)
results = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)
fitness = results.metrics['f1_score']
```

**For Cluster Execution:**
```bash
# experiments/configs/my_experiment.py defines config object
python experiments/run_experiment.py experiments.configs.my_experiment
```

**For Notebooks:**
```python
from qsvm.config.defaults import default_shot_based_config
from qsvm.models import QSVM
from qsvm.data import DataPipeline

pipeline = DataPipeline.from_config(config.data)
x_train, y_train, x_test, y_test = pipeline.load_and_split()

qsvm = QSVM.from_config(default_shot_based_config)
results = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)
print(results.to_dict())
```

### Success Criteria

#### Automated Verification:
- [ ] All modules import without errors: `python -c "import qsvm"`
- [ ] Shot-based kernel produces results: `python experiments/run_experiment.py experiments.configs.shot_sweep`
- [ ] Statevector kernel produces results: Test statevector config
- [ ] Results saved as valid JSON: `python -c "import json; json.load(open('results/experiment.json'))"`
- [ ] Notebooks execute end-to-end: Run notebook cells

#### Manual Verification:
- [ ] Can create new experiment config in <5 lines of Python
- [ ] Can swap kernel strategy by changing one config parameter
- [ ] Can swap feature map by changing one config parameter
- [ ] Results match original implementation (spot check accuracy values)
- [ ] Parallel execution scales on 72-core cluster
- [ ] JSON results are human-readable and contain all relevant info

## What We're NOT Doing

- No unit tests (research code prioritizes velocity)
- No backward compatibility with shots-results.txt format
- No YAML configuration files (Python dataclasses only)
- No disk caching for kernel matrices (may add later)
- No configuration validation (assume valid inputs)
- No visualization tools (separate concern, can use notebooks)
- No database storage (JSON files sufficient)
- No deployment/production concerns (research environment only)
- No support for datasets other than SUSY initially (but architecture allows easy addition)

## Implementation Approach

### Strategy Pattern Application

The core insight is that both kernel implementations follow the Strategy pattern:

**Context**: `QSVM` class orchestrates the workflow
**Strategy Interface**: `QuantumKernel` abstract base class
**Concrete Strategies**: `ShotBasedKernel`, `StatevectorKernel`

This allows genetic algorithms to swap strategies, vary parameters, and compose experiments without touching core code.

### Configuration Design Philosophy

Dataclasses provide:
- Type hints for IDE support
- Immutability (frozen dataclasses)
- Easy serialization/deserialization
- Programmatic construction (GA-friendly)
- No parsing overhead

### Parallelization Strategy

All parallel execution uses `concurrent.futures.ProcessPoolExecutor`:
- Shot-based kernel: Parallelize kernel matrix rows (current approach)
- Keep thread limits in worker processes (`os.environ` settings)
- Configure worker count based on available cores (default: 72 for cluster)

## Phase 1: Core Abstractions and Configuration

### Overview
Establish the foundational architecture: base classes, configuration system, and type definitions. This phase creates the skeleton that all other components will use.

### Changes Required

#### 1. Configuration Module

**File**: `qsvm/config/__init__.py`
```python
from .types import (
    ExperimentConfig,
    KernelConfig,
    FeatureMapConfig,
    DataConfig,
    SVMConfig,
    MetricsResult,
)
from .defaults import (
    default_shot_based_config,
    default_statevector_config,
)

__all__ = [
    "ExperimentConfig",
    "KernelConfig",
    "FeatureMapConfig",
    "DataConfig",
    "SVMConfig",
    "MetricsResult",
    "default_shot_based_config",
    "default_statevector_config",
]
```

**File**: `qsvm/config/types.py`
```python
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Dict, Any
import numpy as np

@dataclass(frozen=True)
class FeatureMapConfig:
    """Configuration for quantum feature map."""
    type: Literal["z", "zz", "pauli"]
    feature_dimension: int
    reps: int
    paulis: Optional[list[str]] = None  # Only for pauli type

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

    # Shot-based specific
    shots: Optional[int] = 2048
    optimization_level: Optional[int] = 1
    workers: Optional[int] = 20
    show_progress: Optional[bool] = False

    # Statevector specific (none currently, but reserved for future)
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

    # Preprocessing
    scale_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4)
    scale_factor: float = 1.0  # Additional multiplicative scaling (e.g., 2**-1)

    # Splitting
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

    # Metadata
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
    kernel_train: Optional[np.ndarray] = None  # Optional: save kernel matrix

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'predictions': self.predictions.tolist(),
            'kernel_shape': self.kernel_train.shape if self.kernel_train is not None else None,
        }
```

**File**: `qsvm/config/defaults.py`
```python
import numpy as np
from .types import (
    ExperimentConfig,
    KernelConfig,
    FeatureMapConfig,
    DataConfig,
    SVMConfig,
)

# Default shot-based configuration matching original shots.py
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

# Default statevector configuration matching notebook experiments
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
        scale_factor=0.5,  # 2**-1
        train_size=500,
        test_size=500,
    ),
    svm=SVMConfig(C=1.0),
)

# Configuration for shot count sweep experiment
shot_sweep_base = default_shot_based_config
```

#### 2. Kernel Base Abstraction

**File**: `qsvm/kernels/__init__.py`
```python
from .base import QuantumKernel
from .shot_based import ShotBasedKernel
from .statevector import StatevectorKernel

__all__ = ["QuantumKernel", "ShotBasedKernel", "StatevectorKernel"]
```

**File**: `qsvm/kernels/base.py`
```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class QuantumKernel(ABC):
    """
    Abstract base class for quantum kernel computation strategies.

    Implements the Strategy pattern for different quantum kernel
    computation approaches (shot-based sampling vs statevector simulation).
    """

    @abstractmethod
    def compute_kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute quantum kernel matrix K[i,j] = kernel(A[i], B[j]).

        Args:
            A: Data matrix of shape (n_samples_a, n_features)
            B: Data matrix of shape (n_samples_b, n_features)

        Returns:
            Kernel matrix of shape (n_samples_a, n_samples_b)
        """
        pass

    @abstractmethod
    def compute_element(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute single kernel element kernel(x, y).

        Args:
            x: Single data point of shape (n_features,)
            y: Single data point of shape (n_features,)

        Returns:
            Kernel value (float)
        """
        pass
```

#### 3. Feature Map Factory

**File**: `qsvm/feature_maps/__init__.py`
```python
from .factory import create_feature_map

__all__ = ["create_feature_map"]
```

**File**: `qsvm/feature_maps/factory.py`
```python
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qsvm.config.types import FeatureMapConfig

def create_feature_map(config: FeatureMapConfig):
    """
    Create Qiskit feature map from configuration.

    Args:
        config: FeatureMapConfig specifying feature map type and parameters

    Returns:
        Qiskit feature map circuit

    Raises:
        ValueError: If feature map type is not recognized
    """
    if config.type == "z":
        return ZFeatureMap(
            feature_dimension=config.feature_dimension,
            reps=config.reps,
        )
    elif config.type == "zz":
        return ZZFeatureMap(
            feature_dimension=config.feature_dimension,
            reps=config.reps,
        )
    elif config.type == "pauli":
        paulis = config.paulis or ['X', 'Y', 'Z']
        return PauliFeatureMap(
            feature_dimension=config.feature_dimension,
            reps=config.reps,
            paulis=paulis,
        )
    else:
        raise ValueError(f"Unknown feature map type: {config.type}")
```

#### 4. Main Package Init

**File**: `qsvm/__init__.py`
```python
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
```

### Success Criteria

#### Automated Verification:
- [ ] Module imports successfully: `python -c "import qsvm"`
- [ ] All config classes importable: `python -c "from qsvm.config import ExperimentConfig"`
- [ ] Default configs are valid: `python -c "from qsvm.config import default_shot_based_config; print(default_shot_based_config)"`
- [ ] Feature map factory imports: `python -c "from qsvm.feature_maps import create_feature_map"`
- [ ] Config serializes to dict: `python -c "from qsvm.config import default_shot_based_config; print(default_shot_based_config.to_dict())"`

#### Manual Verification:
- [ ] Config dataclasses are frozen (immutable)
- [ ] Config to_dict() produces valid JSON-serializable output
- [ ] Feature map factory handles all three types (z, zz, pauli)
- [ ] Type hints work in IDE (autocomplete, error detection)

---

## Phase 2: Data Pipeline

### Overview
Create reusable data loading and preprocessing pipeline that handles the SUSY dataset consistently across all experiments.

### Changes Required

#### 1. Data Pipeline Module

**File**: `qsvm/data/__init__.py`
```python
from .pipeline import DataPipeline

__all__ = ["DataPipeline"]
```

**File**: `qsvm/data/pipeline.py`
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from qsvm.config.types import DataConfig

class DataPipeline:
    """
    Data loading and preprocessing pipeline for quantum SVM experiments.

    Handles:
    - Loading compressed CSV data
    - Feature/target selection
    - Missing value handling
    - MinMaxScaler normalization
    - Additional scaling transformations
    - Train/test splitting
    """

    def __init__(self, config: DataConfig):
        """
        Initialize data pipeline with configuration.

        Args:
            config: DataConfig specifying data source and preprocessing
        """
        self.config = config
        self.scaler = MinMaxScaler(feature_range=config.scale_range)
        self.data_raw = None
        self.data_features = None
        self.data_target = None

    @classmethod
    def from_config(cls, config: DataConfig) -> "DataPipeline":
        """Create pipeline from configuration."""
        return cls(config)

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and clean raw data.

        Returns:
            Tuple of (features_df, target_array)
        """
        # Load CSV
        self.data_raw = pd.read_csv(
            self.config.data_path,
            nrows=self.config.nrows,
            header=None,
        )

        # Drop missing values
        cols_to_check = [self.config.target_column] + list(self.config.feature_columns)
        self.data_raw = self.data_raw.dropna(subset=cols_to_check)

        return self.data_raw

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data: scale features and extract target.

        Returns:
            Tuple of (features_array, target_array)
        """
        if self.data_raw is None:
            self.load_data()

        # Extract and scale features
        features_raw = self.data_raw[list(self.config.feature_columns)]
        self.data_features = self.scaler.fit_transform(features_raw)

        # Apply additional scaling factor
        self.data_features = self.data_features * self.config.scale_factor

        # Extract target
        self.data_target = self.data_raw[self.config.target_column].to_numpy()

        return self.data_features, self.data_target

    def split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.

        Returns:
            Tuple of (x_train, y_train, x_test, y_test)
        """
        if self.data_features is None or self.data_target is None:
            self.preprocess()

        # Train: first N samples
        x_train = self.data_features[:self.config.train_size]
        y_train = self.data_target[:self.config.train_size]

        # Test: last N samples
        x_test = self.data_features[-self.config.test_size:]
        y_test = self.data_target[-self.config.test_size:]

        return x_train, y_train, x_test, y_test

    def load_and_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method: load, preprocess, and split in one call.

        Returns:
            Tuple of (x_train, y_train, x_test, y_test)
        """
        self.load_data()
        self.preprocess()
        return self.split()

    def get_feature_range(self) -> Tuple[float, float]:
        """Get actual feature range after preprocessing."""
        if self.data_features is None:
            self.preprocess()
        return float(self.data_features.min()), float(self.data_features.max())
```

### Success Criteria

#### Automated Verification:
- [ ] Pipeline imports: `python -c "from qsvm.data import DataPipeline"`
- [ ] Pipeline loads SUSY data: Test with default config
- [ ] Feature scaling produces correct range: Check min/max values
- [ ] Train/test split produces correct sizes: Check array shapes

#### Manual Verification:
- [ ] Loaded data matches original implementation (compare first 10 samples)
- [ ] Scaled features in expected range
- [ ] No data leakage between train/test

---

## Phase 3: Shot-Based Kernel Implementation

### Overview
Refactor existing shot-based quantum kernel from shots.py into modular implementation following QuantumKernel interface.

### Changes Required

#### 1. Shot-Based Kernel

**File**: `qsvm/kernels/shot_based.py`
```python
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from tqdm import tqdm

from .base import QuantumKernel
from qsvm.config.types import KernelConfig
from qsvm.feature_maps import create_feature_map

# Set environment variables to control threading
for _env in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_env, "1")


def _compute_row_worker(
    i: int,
    a_vec: np.ndarray,
    B: np.ndarray,
    d: int,
    reps: int,
    shots: int,
    optimization_level: int,
    symmetric: bool,
    feature_map_config: dict,
) -> Tuple[int, List[Tuple[int, float]]]:
    """
    Worker function for parallel kernel row computation.

    Computes one row of the kernel matrix by creating quantum circuits
    that measure overlap between quantum states.

    Args:
        i: Row index
        a_vec: Data point for row i
        B: All data points for columns
        d: Feature dimension (number of qubits)
        reps: Number of feature map repetitions
        shots: Number of quantum measurements
        optimization_level: Qiskit transpiler optimization level
        symmetric: Whether to exploit symmetry (only compute upper triangle)
        feature_map_config: Serialized feature map configuration

    Returns:
        Tuple of (row_index, list of (col_index, kernel_value) pairs)
    """
    # Create simulator with single-threaded execution
    sim = AerSimulator()
    sim.set_options(
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )

    # Create parameterized circuit
    x_params = ParameterVector("x", d)
    y_params = ParameterVector("y", d)

    # Build overlap circuit: Φ(y)† Φ(x) |0⟩
    from qskit.feature_maps import create_feature_map
    from qsvm.config.types import FeatureMapConfig
    fm_config = FeatureMapConfig(**feature_map_config)
    fm = create_feature_map(fm_config)

    qc = QuantumCircuit(d, d)
    qc.compose(fm.assign_parameters(y_params), range(d), inplace=True)
    qc.compose(fm.assign_parameters(x_params).inverse(), range(d), inplace=True)
    qc.measure(range(d), range(d))

    # Transpile once for this circuit structure
    tbase = transpile(qc, sim, optimization_level=optimization_level)

    # Compute kernel values
    j0 = i if symmetric else 0
    out: List[Tuple[int, float]] = []
    zeros = "0" * d

    for j in range(j0, len(B)):
        b_vec = B[j]

        # Bind parameters
        bind_map = {**dict(zip(x_params, a_vec)), **dict(zip(y_params, b_vec))}
        bound = tbase.assign_parameters(bind_map)

        # Execute circuit
        res = sim.run(bound, shots=shots).result()
        counts = res.get_counts()

        # Kernel value = probability of measuring all zeros
        prob_zeros = counts.get(zeros, 0) / shots
        out.append((j, prob_zeros))

    return i, out


class ShotBasedKernel(QuantumKernel):
    """
    Shot-based quantum kernel using AerSimulator with measurements.

    Computes quantum kernel K(x,y) = |⟨Φ(y)|Φ(x)⟩|² by constructing
    overlap circuits and measuring the probability of all-zeros outcome.

    Uses parallel processing to distribute kernel matrix computation
    across multiple worker processes.
    """

    def __init__(self, feature_map_config, kernel_config: KernelConfig):
        """
        Initialize shot-based kernel.

        Args:
            feature_map_config: FeatureMapConfig for quantum encoding
            kernel_config: KernelConfig with shot-based parameters
        """
        self.feature_map_config = feature_map_config
        self.kernel_config = kernel_config
        self.feature_map = create_feature_map(feature_map_config)
        self.d = self.feature_map.num_qubits

        # Extract reps from feature map if available
        try:
            self.reps = int(getattr(self.feature_map, "reps", 2))
        except Exception:
            self.reps = 2

    @classmethod
    def from_configs(cls, feature_map_config, kernel_config: KernelConfig):
        """Create kernel from configurations."""
        return cls(feature_map_config, kernel_config)

    def compute_element(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute single kernel element (not optimized for single calls).

        Args:
            x: Data point (n_features,)
            y: Data point (n_features,)

        Returns:
            Kernel value K(x, y)
        """
        # Use compute_kernel for single element (not efficient, but correct)
        K = self.compute_kernel(x.reshape(1, -1), y.reshape(1, -1))
        return float(K[0, 0])

    def compute_kernel(self, A: np.ndarray, B: np.ndarray, symmetric: bool = False) -> np.ndarray:
        """
        Compute quantum kernel matrix using parallel shot-based simulation.

        Args:
            A: Data matrix (n_samples_a, n_features)
            B: Data matrix (n_samples_b, n_features)
            symmetric: If True, assumes A==B and only computes upper triangle

        Returns:
            Kernel matrix K of shape (n_samples_a, n_samples_b)
        """
        A = np.asarray(A, float)
        B = np.asarray(B, float)

        if symmetric:
            if len(A) != len(B):
                raise ValueError("symmetric=True requires A and B to have same length")

        K = np.zeros((len(A), len(B)), dtype=float)

        # Serialize feature map config for workers
        fm_config_dict = self.feature_map_config.to_dict()

        # Parallel computation
        tasks = []
        with ProcessPoolExecutor(max_workers=self.kernel_config.workers) as ex:
            for i, a in enumerate(A):
                tasks.append(
                    ex.submit(
                        _compute_row_worker,
                        i,
                        np.asarray(a, float),
                        B,
                        self.d,
                        self.reps,
                        int(self.kernel_config.shots),
                        int(self.kernel_config.optimization_level),
                        bool(symmetric),
                        fm_config_dict,
                    )
                )

            # Collect results
            iterator = as_completed(tasks)
            if self.kernel_config.show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc="Computing kernel rows", leave=False)

            for fut in iterator:
                i, pairs = fut.result()
                if symmetric:
                    # Fill both K[i,j] and K[j,i]
                    for j, v in pairs:
                        K[i, j] = v
                        K[j, i] = v
                else:
                    # Fill K[i,j] only
                    for j, v in pairs:
                        K[i, j] = v

        return K
```

### Success Criteria

#### Automated Verification:
- [ ] Kernel imports: `python -c "from qsvm.kernels import ShotBasedKernel"`
- [ ] Kernel computes matrix: Test on small dummy data (10x10)
- [ ] Parallel execution works: Check ProcessPoolExecutor runs
- [ ] Symmetric optimization works: Verify upper triangle computation

#### Manual Verification:
- [ ] Kernel values match original shots.py implementation (spot check)
- [ ] Parallel execution scales (time decreases with more workers)
- [ ] Memory usage reasonable (no leaks in worker processes)

---

## Phase 4: Statevector Kernel Implementation

### Overview
Refactor notebook statevector implementation into modular StatevectorKernel class.

### Changes Required

#### 1. Statevector Kernel

**File**: `qsvm/kernels/statevector.py`
```python
import numpy as np
from qiskit.quantum_info import Statevector
from tqdm import tqdm
from typing import Dict

from .base import QuantumKernel
from qsvm.config.types import KernelConfig
from qsvm.feature_maps import create_feature_map


class StatevectorKernel(QuantumKernel):
    """
    Statevector-based quantum kernel using exact simulation.

    Computes quantum kernel K(x,y) = |⟨Φ(y)|Φ(x)⟩|² by explicitly
    computing quantum statevectors and taking inner products.

    Includes caching mechanism to avoid redundant statevector computation.
    """

    def __init__(self, feature_map_config, kernel_config: KernelConfig):
        """
        Initialize statevector kernel.

        Args:
            feature_map_config: FeatureMapConfig for quantum encoding
            kernel_config: KernelConfig (cache_statevectors flag)
        """
        self.feature_map_config = feature_map_config
        self.kernel_config = kernel_config
        self.feature_map = create_feature_map(feature_map_config)
        self.d = self.feature_map.num_qubits

        # Cache for computed statevectors
        self.cache: Dict[tuple, np.ndarray] = {}

    @classmethod
    def from_configs(cls, feature_map_config, kernel_config: KernelConfig):
        """Create kernel from configurations."""
        return cls(feature_map_config, kernel_config)

    def _compute_statevector(self, x: np.ndarray) -> np.ndarray:
        """
        Compute quantum statevector for data point x.

        Args:
            x: Data point (n_features,)

        Returns:
            Statevector data (complex array)
        """
        qc = self.feature_map.assign_parameters(x)
        return Statevector.from_instruction(qc).data

    def _resolve_statevectors(self, mat: np.ndarray) -> np.ndarray:
        """
        Resolve statevectors for all data points, using cache if enabled.

        Args:
            mat: Data matrix (n_samples, n_features)

        Returns:
            Statevector matrix (2**n_qubits, n_samples)
        """
        out_cols = []

        iterator = mat
        if self.kernel_config.show_progress:
            iterator = tqdm(mat, desc="Computing statevectors", leave=False)

        for row in iterator:
            if self.kernel_config.cache_statevectors:
                # Use cache
                key = tuple(row)
                if key not in self.cache:
                    self.cache[key] = self._compute_statevector(row)
                out_cols.append(self.cache[key])
            else:
                # No caching
                out_cols.append(self._compute_statevector(row))

        return np.column_stack(out_cols)

    def compute_element(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute single kernel element.

        Args:
            x: Data point (n_features,)
            y: Data point (n_features,)

        Returns:
            Kernel value K(x, y)
        """
        psi_x = self._compute_statevector(x)
        psi_y = self._compute_statevector(y)

        # Kernel = |⟨ψ_y|ψ_x⟩|²
        inner_product = np.dot(psi_y.conj(), psi_x)
        return float(np.abs(inner_product) ** 2)

    def compute_kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute quantum kernel matrix using statevector simulation.

        Args:
            A: Data matrix (n_samples_a, n_features)
            B: Data matrix (n_samples_b, n_features)

        Returns:
            Kernel matrix K of shape (n_samples_a, n_samples_b)
        """
        A = np.asarray(A, float)
        B = np.asarray(B, float)

        # Compute statevector matrices
        PsiA = self._resolve_statevectors(A)  # (2**d, n_samples_a)
        PsiB = self._resolve_statevectors(B)  # (2**d, n_samples_b)

        # Gram matrix: G[i,j] = ⟨ψ_A[i]|ψ_B[j]⟩
        G = PsiA.conj().T @ PsiB  # (n_samples_a, n_samples_b)

        # Kernel matrix: K[i,j] = |G[i,j]|²
        K = np.abs(G) ** 2

        return K

    def clear_cache(self):
        """Clear statevector cache."""
        self.cache.clear()
```

### Success Criteria

#### Automated Verification:
- [ ] Kernel imports: `python -c "from qsvm.kernels import StatevectorKernel"`
- [ ] Kernel computes matrix: Test on small dummy data
- [ ] Caching works: Verify cache hit on repeated data points
- [ ] Results are deterministic: Same input produces same output

#### Manual Verification:
- [ ] Kernel values match notebook implementation (spot check)
- [ ] Cache reduces computation time for repeated data
- [ ] Memory usage acceptable (cache doesn't grow unbounded)

---

## Phase 5: QSVM Orchestrator

### Overview
Create main QSVM class that orchestrates the complete workflow using strategy pattern for kernel selection.

### Changes Required

#### 1. QSVM Model

**File**: `qsvm/models/__init__.py`
```python
from .qsvm import QSVM

__all__ = ["QSVM"]
```

**File**: `qsvm/models/qsvm.py`
```python
import numpy as np
from sklearn import svm
from typing import Optional

from qsvm.config.types import ExperimentConfig, ExperimentResult, MetricsResult
from qsvm.kernels import QuantumKernel, ShotBasedKernel, StatevectorKernel
from qsvm.feature_maps import create_feature_map


class QSVM:
    """
    Quantum Support Vector Machine classifier.

    Orchestrates the quantum kernel SVM workflow:
    1. Compute quantum kernel matrix for training data
    2. Train classical SVM with precomputed kernel
    3. Compute quantum kernel for test data
    4. Predict using trained SVM

    Uses Strategy pattern to swap between different quantum kernel
    computation methods (shot-based vs statevector).
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize QSVM with experiment configuration.

        Args:
            config: Complete experiment configuration
        """
        self.config = config

        # Create quantum kernel based on strategy
        if config.kernel.strategy == "shot_based":
            self.kernel = ShotBasedKernel.from_configs(
                config.feature_map,
                config.kernel,
            )
        elif config.kernel.strategy == "statevector":
            self.kernel = StatevectorKernel.from_configs(
                config.feature_map,
                config.kernel,
            )
        else:
            raise ValueError(f"Unknown kernel strategy: {config.kernel.strategy}")

        # SVM components (set during training)
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.kernel_matrix_train: Optional[np.ndarray] = None
        self.svm_model: Optional[svm.SVC] = None

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "QSVM":
        """Create QSVM from configuration."""
        return cls(config)

    def compute_train_kernel(self, x_train: np.ndarray) -> np.ndarray:
        """
        Compute quantum kernel matrix for training data.

        Args:
            x_train: Training features (n_train, n_features)

        Returns:
            Kernel matrix (n_train, n_train)
        """
        self.x_train = np.asarray(x_train, float)

        # Exploit symmetry for training kernel
        if hasattr(self.kernel, 'compute_kernel'):
            # Shot-based kernel supports symmetric optimization
            if isinstance(self.kernel, ShotBasedKernel):
                self.kernel_matrix_train = self.kernel.compute_kernel(
                    self.x_train,
                    self.x_train,
                    symmetric=True,
                )
            else:
                self.kernel_matrix_train = self.kernel.compute_kernel(
                    self.x_train,
                    self.x_train,
                )
        else:
            raise NotImplementedError("Kernel must implement compute_kernel method")

        return self.kernel_matrix_train

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train SVM with quantum kernel.

        Args:
            x_train: Training features (n_train, n_features)
            y_train: Training labels (n_train,)
        """
        self.y_train = np.asarray(y_train)

        # Compute kernel matrix if not already computed
        if self.kernel_matrix_train is None or self.x_train is not x_train:
            self.compute_train_kernel(x_train)

        # Create and train SVM
        self.svm_model = svm.SVC(
            kernel=self.config.svm.kernel,
            C=self.config.svm.C,
        )
        self.svm_model.fit(self.kernel_matrix_train, self.y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for test data.

        Args:
            x_test: Test features (n_test, n_features)

        Returns:
            Predicted labels (n_test,)
        """
        if self.svm_model is None:
            raise RuntimeError("Must call train() before predict()")

        x_test = np.asarray(x_test, float)

        # Compute test kernel: K(test, train)
        kernel_matrix_test = self.kernel.compute_kernel(x_test, self.x_train)

        # Predict using SVM
        predictions = self.svm_model.predict(kernel_matrix_test)

        return predictions

    def fit_predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
    ) -> np.ndarray:
        """
        Train and predict in one call.

        Args:
            x_train: Training features
            y_train: Training labels
            x_test: Test features

        Returns:
            Predicted labels for test data
        """
        self.train(x_train, y_train)
        return self.predict(x_test)

    def fit_evaluate(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ExperimentResult:
        """
        Train, predict, and evaluate in one call.

        Args:
            x_train: Training features
            y_train: Training labels
            x_test: Test features
            y_test: True test labels

        Returns:
            ExperimentResult with metrics and predictions
        """
        from qsvm.evaluation import evaluate_predictions

        # Train and predict
        predictions = self.fit_predict(x_train, y_train, x_test)

        # Evaluate
        metrics = evaluate_predictions(y_test, predictions)

        # Create result object
        result = ExperimentResult(
            config=self.config,
            metrics=metrics,
            predictions=predictions,
            kernel_train=self.kernel_matrix_train,
        )

        return result
```

### Success Criteria

#### Automated Verification:
- [ ] QSVM imports: `python -c "from qsvm import QSVM"`
- [ ] Can create QSVM from config: Test with default configs
- [ ] Can train and predict: Test on dummy data
- [ ] Strategy switching works: Test both shot-based and statevector configs

#### Manual Verification:
- [ ] Predictions match original implementation
- [ ] Kernel strategy swap works correctly
- [ ] Workflow is intuitive (fit_evaluate is easy to use)

---

## Phase 6: Evaluation and Metrics

### Overview
Create metrics evaluation and results persistence module with JSON output.

### Changes Required

#### 1. Evaluation Module

**File**: `qsvm/evaluation/__init__.py`
```python
from .metrics import evaluate_predictions, save_results, load_results

__all__ = ["evaluate_predictions", "save_results", "load_results"]
```

**File**: `qsvm/evaluation/metrics.py`
```python
import json
import numpy as np
from pathlib import Path
from sklearn import metrics
from datetime import datetime
from typing import Union

from qsvm.config.types import MetricsResult, ExperimentResult


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsResult:
    """
    Compute classification metrics.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)

    Returns:
        MetricsResult with accuracy, precision, recall, f1_score
    """
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)

    return MetricsResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def save_results(
    result: ExperimentResult,
    output_path: Union[str, Path],
    include_predictions: bool = True,
    include_kernel: bool = False,
) -> None:
    """
    Save experiment results to JSON file.

    Args:
        result: ExperimentResult to save
        output_path: Path to output JSON file
        include_predictions: Whether to include prediction array
        include_kernel: Whether to include kernel matrix (can be large)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    result_dict = result.to_dict()

    # Optionally remove large arrays
    if not include_predictions:
        result_dict.pop('predictions', None)
    if not include_kernel:
        result_dict.pop('kernel_shape', None)

    # Add metadata
    result_dict['timestamp'] = datetime.now().isoformat()
    result_dict['qsvm_version'] = "2.0.0"

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)


def load_results(input_path: Union[str, Path]) -> dict:
    """
    Load experiment results from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary with experiment results
    """
    with open(input_path, 'r') as f:
        return json.load(f)


class ResultsWriter:
    """
    Utility for writing multiple experiment results to a single JSON file.

    Useful for parameter sweeps and batch experiments.
    """

    def __init__(self, output_path: Union[str, Path]):
        """
        Initialize results writer.

        Args:
            output_path: Path to output JSON file
        """
        self.output_path = Path(output_path)
        self.results = []

    def add_result(self, result: ExperimentResult) -> None:
        """Add experiment result to collection."""
        self.results.append(result.to_dict())

    def save(self) -> None:
        """Save all results to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'timestamp': datetime.now().isoformat(),
            'qsvm_version': "2.0.0",
            'n_experiments': len(self.results),
            'results': self.results,
        }

        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
```

### Success Criteria

#### Automated Verification:
- [ ] Evaluation imports: `python -c "from qsvm.evaluation import evaluate_predictions"`
- [ ] Metrics computed correctly: Test on dummy predictions
- [ ] JSON serialization works: Test save_results
- [ ] JSON deserialization works: Test load_results
- [ ] ResultsWriter collects multiple results: Test batch writing

#### Manual Verification:
- [ ] JSON output is human-readable
- [ ] JSON contains all relevant information
- [ ] Metrics match sklearn direct computation

---

## Phase 7: Experiment Runner and Config Examples

### Overview
Create generic experiment runner script and example configuration files for cluster execution.

### Changes Required

#### 1. Generic Experiment Runner

**File**: `experiments/__init__.py`
```python
# Empty file to make experiments a package
```

**File**: `experiments/run_experiment.py`
```python
#!/usr/bin/env python3
"""
Generic experiment runner for QSVM experiments.

Usage:
    python experiments/run_experiment.py experiments.configs.my_config
    python experiments/run_experiment.py experiments.configs.shot_sweep --output results/
"""

import sys
import argparse
import importlib
from pathlib import Path

from qsvm import QSVM
from qsvm.data import DataPipeline
from qsvm.evaluation import save_results


def run_experiment(config_module: str, output_dir: str = "results"):
    """
    Run experiment from configuration module.

    Args:
        config_module: Python module path to config (e.g., "experiments.configs.my_config")
        output_dir: Directory to save results
    """
    # Import config module
    try:
        module = importlib.import_module(config_module)
    except ImportError as e:
        print(f"Error importing config module '{config_module}': {e}")
        sys.exit(1)

    # Get config object
    if not hasattr(module, 'config'):
        print(f"Config module '{config_module}' must define 'config' variable")
        sys.exit(1)

    config = module.config

    # Optional: Get custom output filename
    output_filename = getattr(module, 'output_filename', None)
    if output_filename is None:
        output_filename = f"{config.name or 'experiment'}.json"

    # Load data
    print(f"Loading data from {config.data.data_path}...")
    pipeline = DataPipeline.from_config(config.data)
    x_train, y_train, x_test, y_test = pipeline.load_and_split()

    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
    print(f"Feature range: {pipeline.get_feature_range()}")

    # Create QSVM
    print(f"Creating QSVM with {config.kernel.strategy} kernel...")
    qsvm = QSVM.from_config(config)

    # Run experiment
    print("Computing training kernel...")
    qsvm.compute_train_kernel(x_train)

    print("Training SVM...")
    qsvm.train(x_train, y_train)

    print("Computing test kernel and predicting...")
    predictions = qsvm.predict(x_test)

    print("Evaluating...")
    from qsvm.evaluation import evaluate_predictions
    metrics = evaluate_predictions(y_test, predictions)

    # Print metrics
    print("\nResults:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1_score:.4f}")

    # Save results
    from qsvm.config.types import ExperimentResult
    result = ExperimentResult(
        config=config,
        metrics=metrics,
        predictions=predictions,
        kernel_train=qsvm.kernel_matrix_train,
    )

    output_path = Path(output_dir) / output_filename
    save_results(result, output_path, include_predictions=False, include_kernel=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run QSVM experiment from configuration module"
    )
    parser.add_argument(
        "config_module",
        help="Python module path to config (e.g., experiments.configs.my_config)"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results (default: results/)"
    )

    args = parser.parse_args()

    run_experiment(args.config_module, args.output)


if __name__ == "__main__":
    main()
```

#### 2. Example Configuration Files

**File**: `experiments/configs/__init__.py`
```python
# Empty file to make configs a package
```

**File**: `experiments/configs/shot_sweep.py`
```python
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

# Base configuration
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
        shots=2048,  # Will be overridden
        optimization_level=1,
        workers=72,  # Cluster configuration
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

# For single experiment, use base config
config = base_config
output_filename = "shot_sweep_2048.json"

# For batch experiments (used by batch runner):
shot_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

def get_configs():
    """Generate configs for all shot counts (for batch runner)."""
    from dataclasses import replace

    configs = []
    for shots in shot_counts:
        # Create new config with updated shots
        kernel_config = replace(base_config.kernel, shots=shots)
        exp_config = replace(
            base_config,
            name=f"shot_sweep_{shots}",
            kernel=kernel_config,
        )
        configs.append(exp_config)

    return configs
```

**File**: `experiments/configs/feature_map_comparison.py`
```python
"""
Configuration for feature map comparison experiment.

Compares Z, ZZ, and Pauli feature maps with different repetitions.
"""

import numpy as np
from qsvm.config import (
    ExperimentConfig,
    KernelConfig,
    FeatureMapConfig,
    DataConfig,
    SVMConfig,
)

# Default config: Z feature map with reps=2
config = ExperimentConfig(
    name="feature_map_z_reps2",
    description="Z feature map with 2 repetitions",
    feature_map=FeatureMapConfig(
        type="z",
        feature_dimension=8,
        reps=2,
    ),
    kernel=KernelConfig(
        strategy="statevector",
        cache_statevectors=True,
        show_progress=False,
    ),
    data=DataConfig(
        data_path="SUSY.csv.gz",
        nrows=10_000,
        train_size=500,
        test_size=500,
        scale_range=(-np.pi / 2, np.pi / 2),
        scale_factor=0.5,
    ),
    svm=SVMConfig(C=1.0),
)

output_filename = "feature_map_z_reps2.json"

# Batch configurations
def get_configs():
    """Generate configs for all feature map combinations."""
    from dataclasses import replace

    configs = []

    for fm_type in ["z", "zz", "pauli"]:
        for reps in [1, 2, 4]:
            # Create feature map config
            fm_config = FeatureMapConfig(
                type=fm_type,
                feature_dimension=8,
                reps=reps,
                paulis=['X', 'Y', 'Z'] if fm_type == "pauli" else None,
            )

            # Create experiment config
            exp_config = replace(
                config,
                name=f"feature_map_{fm_type}_reps{reps}",
                description=f"{fm_type.upper()} feature map with {reps} repetitions",
                feature_map=fm_config,
            )

            configs.append(exp_config)

    return configs
```

**File**: `experiments/configs/hyperparameter_search.py`
```python
"""
Configuration for hyperparameter search experiment.

Varies training size and SVM C parameter.
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
    name="hyperparameter_search",
    description="Hyperparameter search: training size and C",
    feature_map=FeatureMapConfig(
        type="z",
        feature_dimension=8,
        reps=2,
    ),
    kernel=KernelConfig(
        strategy="statevector",
        cache_statevectors=True,
        show_progress=False,
    ),
    data=DataConfig(
        data_path="SUSY.csv.gz",
        nrows=100_000,
        train_size=1000,
        test_size=1000,
        scale_range=(-np.pi / 2, np.pi / 2),
        scale_factor=0.5,
    ),
    svm=SVMConfig(C=3.0),
)

config = base_config
output_filename = "hyperparameter_search.json"

# Batch configurations
training_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
C_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

def get_configs():
    """Generate configs for hyperparameter grid."""
    from dataclasses import replace

    configs = []

    # Training size sweep
    for train_size in training_sizes:
        data_config = replace(base_config.data, train_size=train_size)
        exp_config = replace(
            base_config,
            name=f"train_size_{train_size}",
            description=f"Training size {train_size}",
            data=data_config,
        )
        configs.append(exp_config)

    # C parameter sweep
    for C in C_values:
        svm_config = SVMConfig(C=C)
        exp_config = replace(
            base_config,
            name=f"svm_C_{C}",
            description=f"SVM C parameter {C}",
            svm=svm_config,
        )
        configs.append(exp_config)

    return configs
```

#### 3. Batch Experiment Runner

**File**: `experiments/run_batch.py`
```python
#!/usr/bin/env python3
"""
Batch experiment runner for parameter sweeps.

Usage:
    python experiments/run_batch.py experiments.configs.shot_sweep --output results/shot_sweep/
"""

import sys
import argparse
import importlib
from pathlib import Path

from qsvm import QSVM
from qsvm.data import DataPipeline
from qsvm.evaluation import ResultsWriter


def run_batch_experiments(config_module: str, output_dir: str = "results"):
    """
    Run batch experiments from configuration module.

    Args:
        config_module: Python module path to config
        output_dir: Directory to save results
    """
    # Import config module
    try:
        module = importlib.import_module(config_module)
    except ImportError as e:
        print(f"Error importing config module '{config_module}': {e}")
        sys.exit(1)

    # Get batch configs
    if not hasattr(module, 'get_configs'):
        print(f"Config module '{config_module}' must define 'get_configs()' function")
        sys.exit(1)

    configs = module.get_configs()
    print(f"Running {len(configs)} experiments...")

    # Create results writer
    output_path = Path(output_dir) / "batch_results.json"
    writer = ResultsWriter(output_path)

    # Run each experiment
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running: {config.name}")

        # Load data
        pipeline = DataPipeline.from_config(config.data)
        x_train, y_train, x_test, y_test = pipeline.load_and_split()

        # Run experiment
        qsvm = QSVM.from_config(config)
        result = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)

        # Print metrics
        print(f"  Accuracy: {result.metrics.accuracy:.4f}, F1: {result.metrics.f1_score:.4f}")

        # Add to results
        writer.add_result(result)

    # Save all results
    writer.save()
    print(f"\nAll results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch QSVM experiments from configuration module"
    )
    parser.add_argument(
        "config_module",
        help="Python module path to config"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    run_batch_experiments(args.config_module, args.output)


if __name__ == "__main__":
    main()
```

### Success Criteria

#### Automated Verification:
- [ ] Single experiment runs: `python experiments/run_experiment.py experiments.configs.shot_sweep`
- [ ] Batch experiments run: `python experiments/run_batch.py experiments.configs.shot_sweep`
- [ ] Results saved to JSON: Check output files exist
- [ ] JSON is valid: Parse with json.load()

#### Manual Verification:
- [ ] Easy to create new config (copy example and modify)
- [ ] Config syntax is intuitive
- [ ] Results match original implementation
- [ ] Batch runner completes all experiments
- [ ] Output is organized and readable

---

## Phase 8: Notebook Integration and Documentation

### Overview
Create example Jupyter notebooks demonstrating the refactored library usage and document the architecture.

### Changes Required

#### 1. Interactive Demo Notebook

**File**: `notebooks/01_interactive_demo.ipynb`

Create notebook with cells:

```python
# Cell 1: Setup
import numpy as np
import matplotlib.pyplot as plt
from qsvm import QSVM
from qsvm.config import default_shot_based_config, default_statevector_config
from qsvm.data import DataPipeline

# Cell 2: Load data
pipeline = DataPipeline.from_config(default_shot_based_config.data)
x_train, y_train, x_test, y_test = pipeline.load_and_split()

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Feature range: {pipeline.get_feature_range()}")

# Cell 3: Run shot-based experiment
qsvm_shot = QSVM.from_config(default_shot_based_config)
result_shot = qsvm_shot.fit_evaluate(x_train, y_train, x_test, y_test)

print("Shot-based Results:")
print(f"  Accuracy:  {result_shot.metrics.accuracy:.4f}")
print(f"  Precision: {result_shot.metrics.precision:.4f}")
print(f"  Recall:    {result_shot.metrics.recall:.4f}")
print(f"  F1 Score:  {result_shot.metrics.f1_score:.4f}")

# Cell 4: Run statevector experiment
qsvm_sv = QSVM.from_config(default_statevector_config)
result_sv = qsvm_sv.fit_evaluate(x_train, y_train, x_test, y_test)

print("Statevector Results:")
print(f"  Accuracy:  {result_sv.metrics.accuracy:.4f}")
print(f"  Precision: {result_sv.metrics.precision:.4f}")
print(f"  Recall:    {result_sv.metrics.recall:.4f}")
print(f"  F1 Score:  {result_sv.metrics.f1_score:.4f}")

# Cell 5: Custom configuration
from qsvm.config import ExperimentConfig, KernelConfig, FeatureMapConfig
from dataclasses import replace

# Modify default config
custom_config = replace(
    default_shot_based_config,
    name="custom_experiment",
    feature_map=replace(default_shot_based_config.feature_map, reps=4),
    svm=replace(default_shot_based_config.svm, C=3.0),
)

qsvm_custom = QSVM.from_config(custom_config)
result_custom = qsvm_custom.fit_evaluate(x_train, y_train, x_test, y_test)

print("Custom Config Results:")
print(f"  F1 Score: {result_custom.metrics.f1_score:.4f}")

# Cell 6: Save results
from qsvm.evaluation import save_results

save_results(result_shot, "results/notebook_shot_based.json")
save_results(result_sv, "results/notebook_statevector.json")
print("Results saved!")
```

#### 2. Results Analysis Notebook

**File**: `notebooks/02_results_analysis.ipynb`

Create notebook for analyzing JSON results:

```python
# Cell 1: Load results
from qsvm.evaluation import load_results
import json
import pandas as pd

results = load_results("results/batch_results.json")
print(f"Loaded {results['n_experiments']} experiments")

# Cell 2: Convert to DataFrame
records = []
for r in results['results']:
    records.append({
        'name': r['config']['name'],
        'kernel_strategy': r['config']['kernel']['strategy'],
        'feature_map': r['config']['feature_map']['type'],
        'reps': r['config']['feature_map']['reps'],
        'shots': r['config']['kernel'].get('shots'),
        'C': r['config']['svm']['C'],
        'train_size': r['config']['data']['train_size'],
        'accuracy': r['metrics']['accuracy'],
        'precision': r['metrics']['precision'],
        'recall': r['metrics']['recall'],
        'f1_score': r['metrics']['f1_score'],
    })

df = pd.DataFrame(records)
df.head()

# Cell 3: Analyze by parameter
# Example: Shot count vs accuracy
shot_results = df[df['shots'].notna()].sort_values('shots')
plt.figure(figsize=(10, 6))
plt.plot(shot_results['shots'], shot_results['accuracy'], 'o-')
plt.xscale('log')
plt.xlabel('Shot Count')
plt.ylabel('Accuracy')
plt.title('Shot Count vs Classification Accuracy')
plt.grid(True)
plt.show()

# Cell 4: Feature map comparison
feature_map_results = df.groupby(['feature_map', 'reps'])['f1_score'].mean()
print(feature_map_results)
```

#### 3. README Documentation

**File**: `README.md`

```markdown
# Quantum Support Vector Machine (QSVM) v2.0

Modular, extensible implementation of quantum kernel methods for SVM classification. Designed for high-performance experimentation on cluster environments and genetic algorithm optimization.

## Features

- **Strategy Pattern Architecture**: Easily swap quantum kernel computation strategies
- **Multiple Kernel Methods**: Shot-based (AerSimulator) and statevector simulation
- **Flexible Configuration**: Type-safe Python dataclasses for all experiments
- **Cluster Optimized**: Parallel processing for 72-core environments
- **GA-Ready**: Programmatic configuration generation for genetic algorithms
- **JSON Results**: Structured, human-readable experiment results

## Installation

```bash
# Clone repository
git clone <repo-url>
cd quantum-svm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install numpy pandas scikit-learn qiskit qiskit-aer tqdm
```

## Quick Start

### Notebook Usage

```python
from qsvm import QSVM
from qsvm.config import default_shot_based_config
from qsvm.data import DataPipeline

# Load data
pipeline = DataPipeline.from_config(default_shot_based_config.data)
x_train, y_train, x_test, y_test = pipeline.load_and_split()

# Run experiment
qsvm = QSVM.from_config(default_shot_based_config)
result = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)

print(f"Accuracy: {result.metrics.accuracy:.4f}")
print(f"F1 Score: {result.metrics.f1_score:.4f}")
```

### Cluster Script Usage

```bash
# Run single experiment
python experiments/run_experiment.py experiments.configs.shot_sweep

# Run batch experiments
python experiments/run_batch.py experiments.configs.feature_map_comparison --output results/
```

### Creating Custom Configurations

```python
from qsvm.config import ExperimentConfig, KernelConfig, FeatureMapConfig
import numpy as np

config = ExperimentConfig(
    name="my_experiment",
    feature_map=FeatureMapConfig(type="z", feature_dimension=8, reps=2),
    kernel=KernelConfig(strategy="shot_based", shots=2048, workers=72),
    data=DataConfig(train_size=1000, test_size=1000),
    svm=SVMConfig(C=3.0),
)
```

## Architecture

```
qsvm/
├── kernels/         # Quantum kernel strategies (shot-based, statevector)
├── feature_maps/    # Feature map factory
├── data/            # Data loading and preprocessing
├── models/          # QSVM orchestrator
├── config/          # Configuration dataclasses
├── evaluation/      # Metrics and results
└── utils/           # Utilities

experiments/
├── configs/         # Experiment configurations
├── run_experiment.py    # Single experiment runner
└── run_batch.py         # Batch experiment runner
```

## Genetic Algorithm Integration

The modular design makes it easy to use with genetic algorithms:

```python
# GA generates configuration
config = ExperimentConfig(
    kernel=KernelConfig(strategy="shot_based", shots=genome.shots),
    feature_map=FeatureMapConfig(type="z", reps=genome.reps),
    svm=SVMConfig(C=genome.C),
)

# Evaluate fitness
qsvm = QSVM.from_config(config)
result = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)
fitness = result.metrics.f1_score
```

## Configuration Reference

See `experiments/configs/` for examples:
- `shot_sweep.py`: Shot count parameter sweep
- `feature_map_comparison.py`: Compare Z/ZZ/Pauli feature maps
- `hyperparameter_search.py`: Training size and C parameter search

## License

[Your License]

## Citation

If you use this code in your research, please cite:

```
[Citation information]
```
```

#### 4. Requirements File

**File**: `requirements.txt`

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
qiskit>=0.44.0
qiskit-aer>=0.12.0
tqdm>=4.65.0
matplotlib>=3.7.0
jupyter>=1.0.0
```

### Success Criteria

#### Automated Verification:
- [ ] Dependencies install: `pip install -r requirements.txt`
- [ ] Notebooks execute: Run all cells without errors
- [ ] README examples work: Copy-paste code runs successfully

#### Manual Verification:
- [ ] Notebooks are clear and educational
- [ ] README covers all major use cases
- [ ] Examples are copy-pasteable and work
- [ ] Documentation explains architecture clearly

---

## Phase 9: Migration and Validation

### Overview
Migrate original experiments to new architecture and validate that results match.

### Changes Required

#### 1. Create Migration Script

**File**: `scripts/validate_migration.py`

```python
#!/usr/bin/env python3
"""
Validate that refactored implementation produces same results as original.

Runs sample experiments and compares metrics.
"""

import numpy as np
from qsvm import QSVM
from qsvm.config import default_shot_based_config
from qsvm.data import DataPipeline

def main():
    print("Validating migration...")
    print("=" * 60)

    # Load data with original parameters
    print("\n1. Loading data...")
    pipeline = DataPipeline.from_config(default_shot_based_config.data)
    x_train, y_train, x_test, y_test = pipeline.load_and_split()

    print(f"   Train size: {len(x_train)}")
    print(f"   Test size: {len(x_test)}")
    print(f"   Feature range: {pipeline.get_feature_range()}")

    # Run shot-based experiment
    print("\n2. Running shot-based experiment...")
    qsvm = QSVM.from_config(default_shot_based_config)
    result = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)

    print(f"   Accuracy:  {result.metrics.accuracy:.4f}")
    print(f"   Precision: {result.metrics.precision:.4f}")
    print(f"   Recall:    {result.metrics.recall:.4f}")
    print(f"   F1 Score:  {result.metrics.f1_score:.4f}")

    # Expected values from original implementation (approximate)
    expected_accuracy = 0.72  # Ballpark from original experiments
    tolerance = 0.05  # Allow 5% variation due to stochastic nature

    print("\n3. Validation:")
    if abs(result.metrics.accuracy - expected_accuracy) < tolerance:
        print("   ✓ Accuracy within expected range")
    else:
        print(f"   ✗ Accuracy outside expected range (expected ~{expected_accuracy:.2f})")

    print("\n4. Testing JSON serialization...")
    from qsvm.evaluation import save_results, load_results
    save_results(result, "validation_results.json")
    loaded = load_results("validation_results.json")
    print("   ✓ Results saved and loaded successfully")

    print("\n" + "=" * 60)
    print("Validation complete!")

    return result

if __name__ == "__main__":
    main()
```

#### 2. Move Old Files

```bash
mkdir old_implementation
mv shots.py old_implementation/
mv svm_susy_quantum_eval_0.ipynb old_implementation/
mv svm_susy_quantum_eval_1.ipynb old_implementation/
```

### Success Criteria

#### Automated Verification:
- [ ] Validation script runs: `python scripts/validate_migration.py`
- [ ] Metrics within tolerance: Accuracy ~72% ± 5%
- [ ] JSON serialization works: Results save and load
- [ ] Old files archived: Moved to old_implementation/

#### Manual Verification:
- [ ] Results "feel right" compared to original (spot checks)
- [ ] Performance comparable (execution time similar)
- [ ] No obvious bugs or errors

---

## Performance Considerations

### Parallel Execution Scaling

The shot-based kernel is designed for 72-core cluster execution:

**Current configuration** (`shots.py:177`):
- 65 workers for 72-core cluster
- Leaves ~7 cores for OS and overhead
- Each worker runs single-threaded AerSimulator

**Optimization opportunities**:
1. **Dynamic worker allocation**: Adjust based on available cores
2. **Adaptive shots**: Use fewer shots for training, more for testing
3. **Batch size tuning**: Experiment with different ProcessPoolExecutor configurations

### Memory Usage

**Shot-based kernel**:
- Memory efficient: computes one kernel row at a time
- Scales well with large datasets
- Worker memory isolated in separate processes

**Statevector kernel**:
- Memory intensive: 2^n_qubits per state vector
- Cache grows with unique data points
- Limited to ~10-12 qubits on typical hardware

**For 8 qubits**:
- Statevector size: 2^8 = 256 complex numbers = 4 KB
- Cache for 25,600 unique samples: ~100 MB (manageable)

### Genetic Algorithm Integration

**Fast configuration generation**:
```python
# GA can generate thousands of configs per second
configs = [
    create_config(genome) for genome in ga_population
]

# Parallel evaluation across cluster
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=72) as executor:
    futures = [
        executor.submit(evaluate_qsvm, config)
        for config in configs
    ]
    fitnesses = [f.result() for f in futures]
```

**Recommended GA-friendly parameters**:
- Population size: 100-1000
- Generations: 50-100
- Evaluation parallelism: Full cluster (72 workers)
- Cache kernel computations if data doesn't change

## Migration Notes

### Breaking Changes from v1.0

1. **Configuration**: Hard-coded parameters → Dataclass configs
2. **File format**: Tab-separated text → JSON
3. **API**: QCalculator class → QSVM + strategy pattern
4. **Imports**: `from shots import QCalculator` → `from qsvm import QSVM`

### Compatibility

**Not compatible**:
- Old shots-results.txt files (use JSON instead)
- Direct QCalculator instantiation (use QSVM.from_config)

**Compatible**:
- SUSY.csv.gz dataset (same format)
- Qiskit feature maps (same circuits)
- Sklearn SVM models (same algorithms)

### Migration Path

For existing scripts using old implementation:

**Before**:
```python
from shots import QCalculator
from qiskit.circuit.library import z_feature_map

fm = z_feature_map(feature_dimension=8, reps=2)
qc = QCalculator(fm, shots=2048, a_workers=65)
qc.calculate_kernel(x_train)
qc.svm(1.0, y_train)
preds = qc.predict(x_test, y_test)
```

**After**:
```python
from qsvm import QSVM
from qsvm.config import default_shot_based_config

qsvm = QSVM.from_config(default_shot_based_config)
result = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)
preds = result.predictions
```

## References

- Original implementation: `old_implementation/shots.py`
- Research document: `thoughts/shared/research/2025-11-02-quantum-svm-codebase.md`
- Qiskit documentation: https://qiskit.org/documentation/
- Scikit-learn SVM: https://scikit-learn.org/stable/modules/svm.html