import os
import numpy as np
from qiskit.quantum_info import Statevector
from tqdm import tqdm
from typing import Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from .base import QuantumKernel
from qsvm.config.types import KernelConfig
from qsvm.feature_maps import create_feature_map

for _env in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_env, "1")


def _compute_statevector_worker(
    x: np.ndarray,
    feature_map_config: dict,
    is_custom_ansatz: bool,
    custom_genome: Any = None,
    feature_dimension: int = None
) -> Tuple[tuple, np.ndarray]:
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    from qiskit.quantum_info import Statevector
    from qsvm.feature_maps import create_feature_map
    from qsvm.config.types import FeatureMapConfig

    if is_custom_ansatz:
        from qsvm.feature_maps import CustomAnsatz
        feature_map = CustomAnsatz(custom_genome, feature_dimension)
    else:
        fm_config = FeatureMapConfig(**feature_map_config)
        feature_map = create_feature_map(fm_config)

    qc = feature_map.assign_parameters(x)
    sv_data = Statevector.from_instruction(qc).data

    return (tuple(x), sv_data)


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
            kernel_config: KernelConfig (cache_statevectors flag, workers)
        """
        self.feature_map_config = feature_map_config
        self.kernel_config = kernel_config
        self.feature_map = create_feature_map(feature_map_config)
        self.d = self.feature_map.num_qubits
        self.workers = kernel_config.workers or 1

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
        Resolve statevectors for all data points using parallel workers.

        Args:
            mat: Data matrix (n_samples, n_features)

        Returns:
            Statevector matrix (2**n_qubits, n_samples)
        """
        if self.workers == 1 or len(mat) < 10:
            return self._resolve_statevectors_sequential(mat)

        out_dict = {}

        from qsvm.feature_maps import CustomAnsatz
        is_custom_ansatz = isinstance(self.feature_map, CustomAnsatz)

        if is_custom_ansatz:
            custom_genome = self.feature_map.genome
            feature_dimension = self.feature_map.feature_dimension
            feature_map_config = {}
        else:
            custom_genome = None
            feature_dimension = None
            feature_map_config = self.feature_map_config.to_dict()

        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            tasks = []
            for row in mat:
                key = tuple(row)
                if self.kernel_config.cache_statevectors and key in self.cache:
                    out_dict[key] = self.cache[key]
                else:
                    tasks.append(ex.submit(
                        _compute_statevector_worker,
                        row,
                        feature_map_config,
                        is_custom_ansatz,
                        custom_genome,
                        feature_dimension
                    ))

            iterator = as_completed(tasks)
            if self.kernel_config.show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc="Computing statevectors", leave=False)

            for fut in iterator:
                key, sv_data = fut.result()
                out_dict[key] = sv_data
                if self.kernel_config.cache_statevectors:
                    self.cache[key] = sv_data

        out_cols = [out_dict[tuple(row)] for row in mat]
        return np.column_stack(out_cols)

    def _resolve_statevectors_sequential(self, mat: np.ndarray) -> np.ndarray:
        """
        Sequential version of statevector resolution (original implementation).

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
                key = tuple(row)
                if key not in self.cache:
                    self.cache[key] = self._compute_statevector(row)
                out_cols.append(self.cache[key])
            else:
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

        PsiA = self._resolve_statevectors(A)
        PsiB = self._resolve_statevectors(B)

        G = PsiA.conj().T @ PsiB

        K = np.abs(G) ** 2

        return K

    def clear_cache(self):
        """Clear statevector cache."""
        self.cache.clear()
