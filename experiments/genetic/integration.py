from typing import Optional
import numpy as np
from qsvm.models.qsvm import QSVM
from qsvm.config.types import ExperimentConfig, KernelConfig
from qsvm.kernels import StatevectorKernel, ShotBasedKernel
from qsvm.feature_maps.custom_ansatz import CustomAnsatz


def create_qsvm_with_custom_ansatz(
    ansatz: CustomAnsatz,
    config: ExperimentConfig
) -> QSVM:
    qsvm = QSVM.__new__(QSVM)
    qsvm.config = config

    if config.kernel.strategy == "statevector":
        qsvm.kernel = create_statevector_kernel_with_ansatz(ansatz, config.kernel)
    elif config.kernel.strategy == "shot_based":
        qsvm.kernel = create_shot_based_kernel_with_ansatz(ansatz, config.kernel)
    else:
        raise ValueError(f"Unknown kernel strategy: {config.kernel.strategy}")

    qsvm.x_train = None
    qsvm.y_train = None
    qsvm.kernel_matrix_train = None
    qsvm.svm_model = None

    return qsvm


def create_statevector_kernel_with_ansatz(
    ansatz: CustomAnsatz,
    kernel_config: KernelConfig
) -> StatevectorKernel:
    kernel = StatevectorKernel.__new__(StatevectorKernel)
    kernel.feature_map = ansatz
    kernel.feature_map_config = None
    kernel.kernel_config = kernel_config
    kernel.d = ansatz.num_qubits
    kernel.workers = kernel_config.workers or 1
    kernel.cache = {}
    return kernel


def create_shot_based_kernel_with_ansatz(
    ansatz: CustomAnsatz,
    kernel_config: KernelConfig
) -> ShotBasedKernel:
    kernel = ShotBasedKernel.__new__(ShotBasedKernel)
    kernel.feature_map = ansatz
    kernel.feature_map_config = None
    kernel.kernel_config = kernel_config
    kernel.d = ansatz.num_qubits
    try:
        kernel.reps = int(getattr(ansatz, "reps", 2))
    except Exception:
        kernel.reps = 2
    return kernel


def evaluate_ansatz_fitness(
    ansatz: CustomAnsatz,
    config: ExperimentConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> float:
    qsvm = create_qsvm_with_custom_ansatz(ansatz, config)
    result = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)
    return result.metrics.accuracy


class PicklableFitnessFunction:
    """
    Picklable wrapper for fitness function that can be used with multiprocessing.
    """
    def __init__(
        self,
        config: ExperimentConfig,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray
    ):
        self.config = config
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __call__(self, ansatz: CustomAnsatz) -> float:
        return evaluate_ansatz_fitness(
            ansatz=ansatz,
            config=self.config,
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test
        )
