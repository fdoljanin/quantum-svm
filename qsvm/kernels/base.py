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
