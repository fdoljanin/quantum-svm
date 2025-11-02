# Quantum Support Vector Machine (QSVM)

Modular, extensible implementation of quantum kernel methods for SVM classification. Designed for high-performance experimentation on cluster environments.

**Bachelor thesis**: Application of Quantum Support Vector Machine method in data classification
**Author**: Frane Doljanin
**Mentor**: Leandra Vranješ Markić

## Features

- **Strategy Pattern Architecture**: Easily swap quantum kernel computation strategies
- **Multiple Kernel Methods**: Shot-based (AerSimulator) and statevector simulation
- **Flexible Configuration**: Type-safe Python dataclasses for all experiments
- **Cluster Optimized**: Parallel processing for 72-core environments
- **JSON Results**: Structured, human-readable experiment results

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd quantum-svm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Quick Test

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick test (2-5 minutes)
python test_qsvm.py
```

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
python experiments/run_experiment.py experiments.configs.shot_sweep --output results/

# Run batch experiments (parameter sweep)
python experiments/run_batch.py experiments.configs.shot_sweep --output results/shot_sweep/
```

## Architecture

```
qsvm/
├── kernels/         # Quantum kernel strategies (shot-based, statevector)
├── feature_maps/    # Feature map factory (Z, ZZ, Pauli)
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

## Creating Custom Configurations

```python
from qsvm.config import ExperimentConfig, KernelConfig, FeatureMapConfig, DataConfig, SVMConfig
import numpy as np

config = ExperimentConfig(
    name="my_experiment",
    feature_map=FeatureMapConfig(type="z", feature_dimension=8, reps=2),
    kernel=KernelConfig(strategy="shot_based", shots=2048, workers=72),
    data=DataConfig(train_size=1000, test_size=1000),
    svm=SVMConfig(C=3.0),
)
```

## Available Configurations

See `experiments/configs/` for examples:
- `shot_sweep.py`: Shot count parameter sweep (1 to 4096 shots)

## Dataset

This project uses the SUSY dataset for supersymmetric particle classification:
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/279/susy)
- **Size**: 5 million collision events, 18 features
- **Task**: Binary classification (signal vs background)
- **File**: `SUSY.csv.gz` (922 MB compressed)

## Configuration Reference

### Kernel Strategies

**Shot-based**: Uses AerSimulator with measurements
- Faster for smaller shot counts
- Probabilistic kernel estimation
- Optimized for cluster parallelization

**Statevector**: Exact quantum state computation
- Deterministic results
- Includes caching for efficiency
- No shot noise

### Feature Maps

- `z`: Z rotation feature map
- `zz`: ZZ entangling feature map
- `pauli`: Pauli feature map

## Performance Tips

- **Local testing**: Use `workers=8`, `train_size=100-500`, `shots=512`
- **Cluster runs**: Use `workers=72`, `train_size=800-1000`, `shots=2048+`
- **Statevector**: Best for small datasets (< 500 samples), no shot parameter needed
- **Shot-based**: Scales better to larger datasets, adjust shots vs speed tradeoff

## Project Structure

```
.
├── qsvm/                   # Main package
├── experiments/            # Experiment configs and runners
├── notebooks/              # Jupyter notebooks for exploration
├── test_qsvm.py           # Quick validation script
├── SUSY.csv.gz            # Dataset
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Requirements

- Python 3.8+
- Qiskit 0.44+
- scikit-learn 1.3+
- NumPy, pandas, matplotlib
- See `requirements.txt` for full list

## About

This is a research project developed as part of a bachelor thesis on quantum machine learning. For questions or collaboration, please contact the author.
