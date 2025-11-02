/---
date: 2025-11-02T00:00:00+01:00
researcher: Claude Code
git_commit: 748543868415501318ed5f70b6c09fa9009969a3
branch: main
repository: quantum-svm
topic: "Understanding the Quantum SVM Codebase - Functionality and Refactoring Opportunities"
tags: [research, codebase, quantum-machine-learning, qsvm, qiskit, bachelor-thesis]
status: complete
last_updated: 2025-11-02
last_updated_by: Claude Code
---

# Research: Understanding the Quantum SVM Codebase

**Date**: 2025-11-02T00:00:00+01:00
**Researcher**: Claude Code
**Git Commit**: 748543868415501318ed5f70b6c09fa9009969a3
**Branch**: main
**Repository**: quantum-svm

## Research Question

What does this quantum-svm codebase do, and what are the opportunities for refactoring to enable further research?

## Summary

This is a **Quantum Support Vector Machine (QSVM)** implementation developed as a Bachelor thesis project on quantum machine learning. The codebase explores using quantum computing to enhance classical machine learning by computing kernel functions through quantum circuits. It evaluates different quantum feature maps on the SUSY particle physics dataset to classify collision events.

**Core functionality:**
- Encodes classical data into quantum states using various feature maps (Z, ZZ, Pauli)
- Computes quantum kernels by measuring quantum state overlaps
- Trains classical SVMs with quantum-computed kernel matrices
- Evaluates classification performance on particle physics data

**Key finding**: Quantum SVMs achieve ~75% accuracy on the SUSY dataset, with performance dependent on feature map type, repetitions, training set size, and quantum measurement shots.

## Detailed Findings

### Main Implementation: shots.py

The primary production code (`shots.py:1-193`) implements a shot-based quantum kernel calculator:

**`QCalculator` class** (`shots.py:71-147`):
- Core class for quantum kernel computation using quantum circuit measurements
- Configurable parameters:
  - `shots`: Number of quantum measurements (default 2048)
  - `optimization_level`: Circuit optimization level (default 1)
  - `a_workers`: Number of parallel worker processes (default 20)
  - `reps`: Feature map repetitions (default 2)

**Key methods:**
- `quantum_kernel()` (`shots.py:92-132`): Computes kernel matrix K[i,j] between data points
  - Parallelizes computation across rows using ProcessPoolExecutor
  - Supports symmetric optimization for training kernels
  - Returns kernel matrix with quantum-computed similarities

- `calculate_kernel()` (`shots.py:134-136`): Computes training kernel matrix
- `svm()` (`shots.py:138-140`): Trains SVM with precomputed quantum kernel
- `predict()` (`shots.py:142-146`): Predicts on test data

**Worker function** `_compute_row_worker()` (`shots.py:27-68`):
- Executes in separate process for parallelization
- Creates quantum circuit with two feature maps (forward and inverse)
- Measures probability of all-zeros outcome: P(|00...0⟩)
- This probability represents quantum kernel value K(x,y)

**Main execution** (`shots.py:149-192`):
- Experiment varying shot counts: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
- Uses 800 training samples, 1000 test samples
- Records accuracy, precision, recall, F1 metrics
- Results saved to `shots-results.txt`

**Technical details:**
- Feature encoding range: [-π/4, π/4] using MinMaxScaler
- Uses 8 features from SUSY dataset (columns 1-8)
- Target is column 0 (binary classification)
- Thread limitation: Forces single-threaded execution to control parallelism
- AerSimulator configured with max_parallel_threads=1 for worker isolation

### Experimental Notebooks

**`svm_susy_quantum_eval_0.ipynb`**: Initial exploration and comparison

Alternative `QCalculator` implementation:
- Uses **statevector method** instead of shot-based sampling
- Computes exact quantum states: ψ = Statevector.from_instruction(qc)
- Calculates kernel as K = |⟨ψ_i|φ_j⟩|²
- Includes caching mechanism for computed state vectors
- More accurate but memory-intensive approach

Experiments conducted:
1. **Classical baseline**: SVM with RBF kernel achieves 71.8% accuracy
2. **Feature scaling tests**: Tried 8 different scaling factors (2^-i for i=0..7)
3. **Feature map comparison** with various repetitions:
   - Z-feature map (ZFM): reps=1,2,4
   - ZZ-feature map (ZZFM): reps=1,2,4
   - Pauli feature map: reps=1,2,4 with ['X','Y','Z'] gates

Best results: **ZFM with reps=2 at scale 2^-1 achieved 77% accuracy**

**`svm_susy_quantum_eval_1.ipynb`**: Hyperparameter optimization

Additional experiments:
1. **Training set size scaling**:
   - Tested: 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600 samples
   - Performance improves with more training data
   - Best: 75.6% accuracy with 25,600 training samples

2. **C parameter optimization**:
   - Tested C ∈ [0.5, 1, 2, 3, 4, 5]
   - Sweet spot: C=3.0 achieved 75.2% accuracy
   - Higher C values (10+) showed overfitting

3. **Final configuration**:
   - Z-feature map with reps=2
   - Feature scaling: 2^-1 (multiply by 0.5)
   - C=3.0
   - Training size: 1000 samples
   - Test size: 1000 samples

### Data: SUSY Dataset

**`SUSY.csv.gz`** (922 MB compressed):
- Particle physics collision data
- Binary classification: signal (1) vs background (0)
- Features used (columns 1-8): 8-dimensional feature space
- Target (column 0): Binary label
- Dataset size: Code loads up to 100,000 rows for experiments
- Class distribution: Approximately balanced (~54% background)

### Architecture Insights

**Quantum Kernel Computation Strategy**:

The core innovation is computing kernel functions K(x,y) using quantum circuits:

1. **Encoding**: Map classical data x → quantum state |ψ(x)⟩ using feature map U(x)
2. **Overlap measurement**: Compute K(x,y) = |⟨ψ(y)|ψ(x)⟩|²
3. **Circuit implementation**:
   - Apply U(y) to |0⟩ → |ψ(y)⟩
   - Apply U†(x) (inverse) to get U†(x)|ψ(y)⟩
   - Measure in computational basis
   - P(|00...0⟩) gives kernel value

**Two Implementation Approaches**:

| Aspect | Statevector (notebooks) | Shot-based (shots.py) |
|--------|------------------------|----------------------|
| Computation | Exact state calculation | Probabilistic sampling |
| Memory | High (exponential in qubits) | Low (fixed per shot) |
| Accuracy | Perfect | Statistical (~1/√shots) |
| Realism | Ideal quantum computer | Mimics real hardware |
| Speed | Fast for small systems | Scalable with parallelism |

**Parallelization Architecture**:
- Row-level parallelization: Each row of kernel matrix computed by separate worker
- ProcessPoolExecutor manages worker pool (default 20 workers)
- Symmetric optimization: For training kernel K[i,j]=K[j,i], compute upper triangle only
- Environment variables set to force single-threading in numerical libraries
- Prevents oversubscription when multiple workers run simultaneously

**Feature Map Types**:

1. **Z-feature map** (ZFeatureMap):
   - Single-qubit Z-rotations: e^(-iZ x_i)
   - Simple, few gates
   - Best performance in experiments

2. **ZZ-feature map** (ZZFeatureMap):
   - Adds entanglement: e^(-iZZ x_i x_j)
   - Two-qubit interactions
   - More expressive but noisier

3. **Pauli feature map**:
   - Uses X, Y, Z rotations
   - Most general
   - Highest circuit complexity

**Performance Characteristics**:
- Accuracy: 70-77% depending on configuration
- Best: Z-feature map, reps=2, scale=0.5, C=3.0
- Shot count: Diminishing returns above 1024-2048 shots
- Training size: Linear improvement up to 25,600 samples
- Repetitions: Sweet spot at reps=2-3 (more causes overfitting/noise)

### Code Quality Observations

**Strengths**:
- Clean dataclass usage for configuration
- Proper parallel processing with process isolation
- Comprehensive experiments documented in notebooks
- Results systematically recorded

**Issues** (as noted in README):
- Code described as "messy" and "run in ipynb env"
- No dependency specification (requirements.txt missing)
- Hard-coded parameters scattered throughout
- Duplicate implementations (statevector vs shots)
- Minimal error handling
- No logging or progress tracking (except tqdm in notebooks)
- No tests or validation

## Refactoring Opportunities

### 1. Project Structure

**Current state**: Flat structure with scripts and notebooks

**Recommended structure**:
```
quantum-svm/
├── qsvm/                      # Main package
│   ├── __init__.py
│   ├── kernels/               # Kernel computation
│   │   ├── __init__.py
│   │   ├── base.py           # Base kernel class
│   │   ├── statevector.py    # Exact computation
│   │   └── shots.py          # Shot-based computation
│   ├── feature_maps/          # Quantum feature maps
│   │   ├── __init__.py
│   │   └── factory.py        # Feature map creation
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── susy.py           # SUSY dataset handler
│   ├── models/                # SVM models
│   │   ├── __init__.py
│   │   └── qsvm.py           # Main QSVM class
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── metrics.py        # Evaluation metrics
│       └── parallel.py       # Parallelization helpers
├── experiments/               # Experiment scripts
│   ├── shot_experiment.py    # shots.py refactored
│   ├── feature_map_comparison.py
│   └── hyperparameter_search.py
├── notebooks/                 # Analysis notebooks
│   ├── 01_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── tests/                     # Unit tests
│   ├── test_kernels.py
│   └── test_feature_maps.py
├── data/                      # Data directory
│   └── SUSY.csv.gz
├── results/                   # Experiment results
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
└── .gitignore
```

### 2. Unified API Design

**Create base abstractions**:

```python
# qsvm/kernels/base.py
from abc import ABC, abstractmethod

class QuantumKernel(ABC):
    @abstractmethod
    def compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute quantum kernel matrix K[i,j] = k(X[i], Y[j])"""
        pass

    @abstractmethod
    def compute_element(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute single kernel element k(x, y)"""
        pass
```

**Implement concrete classes**:

```python
# qsvm/kernels/shots.py
class ShotBasedKernel(QuantumKernel):
    def __init__(self, feature_map, shots=2048, workers=20, ...):
        self.feature_map = feature_map
        self.shots = shots
        self.workers = workers

# qsvm/kernels/statevector.py
class StatevectorKernel(QuantumKernel):
    def __init__(self, feature_map, cache=True):
        self.feature_map = feature_map
        self.cache = cache
        self.cached_states = {}
```

### 3. Configuration Management

**Replace hard-coded values with configuration**:

```python
# qsvm/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    # Data
    data_path: str = "data/SUSY.csv.gz"
    n_samples: int = 100_000
    features: list[int] = (1, 2, 3, 4, 5, 6, 7, 8)
    target: int = 0

    # Preprocessing
    feature_range: tuple[float, float] = (-np.pi/4, np.pi/4)
    train_size: int = 800
    test_size: int = 1000

    # Quantum
    feature_map_type: str = "z"  # "z", "zz", "pauli"
    feature_map_reps: int = 2
    shots: int = 2048
    optimization_level: int = 1

    # Training
    svm_C: float = 1.0
    svm_kernel: str = "precomputed"

    # Execution
    n_workers: int = 20
    show_progress: bool = True

    # Output
    results_path: str = "results/"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file"""
        pass
```

### 4. Dependency Management

**Create `requirements.txt`**:

```
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Quantum computing
qiskit>=0.44.0
qiskit-aer>=0.12.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Progress tracking
tqdm>=4.65.0

# Development
pytest>=7.4.0
jupyter>=1.0.0
black>=23.0.0
mypy>=1.5.0
```

### 5. CLI Interface

**Add command-line interface**:

```python
# experiments/run_experiment.py
import click
from qsvm.config import ExperimentConfig
from qsvm.models.qsvm import QSVM

@click.command()
@click.option('--config', '-c', type=click.Path(), help='Config YAML file')
@click.option('--shots', type=int, help='Number of quantum shots')
@click.option('--workers', type=int, help='Number of parallel workers')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def run_experiment(config, shots, workers, output):
    """Run QSVM experiment with specified configuration"""
    cfg = ExperimentConfig.from_yaml(config) if config else ExperimentConfig()

    # Override with CLI args
    if shots:
        cfg.shots = shots
    if workers:
        cfg.n_workers = workers
    if output:
        cfg.results_path = output

    # Run experiment
    qsvm = QSVM(cfg)
    results = qsvm.run()
    results.save(cfg.results_path)

if __name__ == '__main__':
    run_experiment()
```

### 6. Logging and Monitoring

**Add proper logging**:

```python
import logging

# qsvm/utils/logging.py
def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
```

### 7. Testing Infrastructure

**Add unit tests**:

```python
# tests/test_kernels.py
import pytest
import numpy as np
from qsvm.kernels.shots import ShotBasedKernel

def test_kernel_symmetry():
    """Test that kernel matrix is symmetric"""
    kernel = ShotBasedKernel(...)
    X = np.random.rand(10, 8)
    K = kernel.compute_kernel(X, X)

    assert np.allclose(K, K.T), "Kernel matrix should be symmetric"

def test_kernel_positive():
    """Test that kernel values are in [0, 1]"""
    kernel = ShotBasedKernel(...)
    X = np.random.rand(10, 8)
    K = kernel.compute_kernel(X, X)

    assert np.all(K >= 0) and np.all(K <= 1), "Kernel values should be in [0,1]"
```

### 8. Visualization Module

**Add plotting utilities**:

```python
# qsvm/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_kernel_matrix(K: np.ndarray, title: str = "Quantum Kernel Matrix"):
    """Visualize kernel matrix as heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(K, cmap='viridis', square=True)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_shot_convergence(results: dict):
    """Plot accuracy vs number of shots"""
    plt.figure(figsize=(10, 6))
    plt.semilogx(results['shots'], results['accuracy'], 'o-')
    plt.xlabel('Number of Shots')
    plt.ylabel('Accuracy')
    plt.title('Shot Count vs Classification Accuracy')
    plt.grid(True)
    return plt.gcf()
```

### 9. Documentation

**Add docstrings and README**:

```python
# qsvm/models/qsvm.py
class QSVM:
    """
    Quantum Support Vector Machine classifier.

    Uses quantum feature maps to compute kernel functions via quantum
    circuit execution. The quantum kernel is then used to train a
    classical SVM classifier.

    Parameters
    ----------
    kernel : QuantumKernel
        Quantum kernel computer (shot-based or statevector)
    C : float, default=1.0
        Regularization parameter for SVM

    Attributes
    ----------
    kernel_matrix_ : ndarray of shape (n_samples, n_samples)
        Computed quantum kernel matrix
    svm_ : sklearn.svm.SVC
        Fitted SVM classifier

    Examples
    --------
    >>> from qsvm.kernels.shots import ShotBasedKernel
    >>> from qiskit.circuit.library import ZFeatureMap
    >>>
    >>> fm = ZFeatureMap(feature_dimension=8, reps=2)
    >>> kernel = ShotBasedKernel(fm, shots=2048)
    >>> qsvm = QSVM(kernel=kernel, C=3.0)
    >>> qsvm.fit(X_train, y_train)
    >>> predictions = qsvm.predict(X_test)
    """
```

### 10. Performance Optimization

**Specific optimizations**:

1. **Kernel caching**: Save computed kernel matrices to disk
2. **Batch processing**: Process multiple test samples simultaneously
3. **Circuit caching**: Cache transpiled circuits for reuse
4. **Adaptive shots**: Use fewer shots for training, more for testing
5. **GPU acceleration**: Leverage Qiskit GPU simulator if available

## Code References

- Main implementation: `shots.py:1-193`
- QCalculator class: `shots.py:71-147`
- Parallel worker: `shots.py:27-68`
- Statevector implementation: `svm_susy_quantum_eval_0.ipynb` cell `65c816c5`
- Shot experiments: `shots.py:149-192`
- Hyperparameter tuning: `svm_susy_quantum_eval_1.ipynb` cells `01088c58`, `40779059`

## Open Questions

1. **Why is performance plateauing at ~75%?**
   - Is it the dataset, feature representation, or quantum circuit limitations?
   - Would real quantum hardware perform differently?

2. **What is the computational cost comparison?**
   - Classical RBF kernel vs quantum kernel computation time
   - Scalability to larger datasets

3. **Can other quantum feature maps improve performance?**
   - Hardware-efficient ansätze
   - Problem-specific feature maps for particle physics

4. **How does noise affect results?**
   - Current experiments use ideal simulator
   - Real quantum hardware has gate errors and decoherence

5. **Is there benefit over classical methods?**
   - RBF kernel achieves similar accuracy
   - Need quantum advantage analysis

## Related Research

- Quantum machine learning literature on QSVMs
- Qiskit machine learning tutorials
- SUSY dataset benchmarks with classical methods

## Recommendations

**Immediate priorities for refactoring**:

1. **Create requirements.txt** - Essential for reproducibility
2. **Modularize shots.py** - Separate concerns into classes
3. **Add CLI interface** - Make experiments easily repeatable
4. **Document configuration** - What parameters affect results

**Medium-term improvements**:

5. **Unify implementations** - Single API for both kernel methods
6. **Add comprehensive tests** - Ensure correctness
7. **Improve logging** - Track experiment progress
8. **Create visualization tools** - Better results analysis

**Long-term research directions**:

9. **Quantum advantage analysis** - Compare to classical carefully
10. **Real hardware experiments** - Test on IBM Quantum
11. **New feature maps** - Explore problem-specific encodings
12. **Other datasets** - Generalize beyond SUSY

The codebase is a solid foundation for quantum machine learning research. With systematic refactoring focusing on modularity, configurability, and documentation, it could become an excellent platform for exploring quantum kernels and advancing quantum ML research.