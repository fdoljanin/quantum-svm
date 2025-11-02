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
    from qsvm.feature_maps import create_feature_map
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
