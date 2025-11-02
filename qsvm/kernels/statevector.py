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
