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
