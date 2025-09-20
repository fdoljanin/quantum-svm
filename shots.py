import os
for _env in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_env, "1")

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple

from sklearn import metrics, svm
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import z_feature_map
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector

from tqdm import tqdm

RELEVANT_FEATURES = [1, 2, 3, 4, 5, 6, 7, 8]
TARGET = 0


# ---------------- worker ----------------
def _compute_row_worker(
    i: int,
    a_vec: np.ndarray,
    B: np.ndarray,
    d: int,
    reps: int,
    shots: int,
    optimization_level: int,
    symmetric: bool,
) -> Tuple[int, List[Tuple[int, float]]]:
    sim = AerSimulator()
    sim.set_options(
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )

    x_params = ParameterVector("x", d)
    y_params = ParameterVector("y", d)

    fm = z_feature_map(feature_dimension=d, reps=reps)
    qc = QuantumCircuit(d, d)
    qc.compose(fm.assign_parameters(y_params), range(d), inplace=True)
    qc.compose(fm.assign_parameters(x_params).inverse(), range(d), inplace=True)
    qc.measure(range(d), range(d))

    tbase = transpile(qc, sim, optimization_level=optimization_level)

    j0 = i if symmetric else 0
    out: List[Tuple[int, float]] = []
    zeros = "0" * d

    for j in range(j0, len(B)):
        b_vec = B[j]
        bind_map = {**dict(zip(x_params, a_vec)), **dict(zip(y_params, b_vec))}
        bound = tbase.assign_parameters(bind_map)
        res = sim.run(bound, shots=shots).result()
        counts = res.get_counts()
        prob_zeros = counts.get(zeros, 0) / shots
        out.append((j, prob_zeros))

    return i, out


@dataclass
class QCalculator:
    fm: z_feature_map
    shots: int = 2048
    optimization_level: int = 1
    show_progress: bool = True
    a_workers: int = 20
    reps: int = 2

    def __post_init__(self):
        self.d = self.fm.num_qubits
        self.x_train = None
        self.kernel = None
        self.svm_linear = None
        self.predictions = None

        try:
            self.reps = int(getattr(self.fm, "reps", self.reps))
        except Exception:
            pass

    def quantum_kernel(self, A: np.ndarray, B: np.ndarray, symmetric: bool = False) -> np.ndarray:
        A = np.asarray(A, float)
        B = np.asarray(B, float)

        if symmetric:
            if len(A) != len(B):
                raise ValueError("symmetric=True requires A and B to have same length")
        K = np.zeros((len(A), len(B)), dtype=float)

        tasks = []
        with ProcessPoolExecutor(max_workers=self.a_workers) as ex:
            for i, a in enumerate(A):
                tasks.append(
                    ex.submit(
                        _compute_row_worker,
                        i,
                        np.asarray(a, float),
                        B,
                        self.d,
                        self.reps,
                        int(self.shots),
                        int(self.optimization_level),
                        bool(symmetric),
                    )
                )

            iterator = as_completed(tasks)
            if self.show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc="A (rows)", leave=False)

            for fut in iterator:
                i, pairs = fut.result()
                if symmetric:
                    for j, v in pairs:
                        K[i, j] = v
                        K[j, i] = v
                else:
                    for j, v in pairs:
                        K[i, j] = v

        return K

    def calculate_kernel(self, x_train: np.ndarray):
        self.x_train = np.asarray(x_train, float)
        self.kernel = self.quantum_kernel(self.x_train, self.x_train, symmetric=True)

    def svm(self, c: float, y_train: np.ndarray):
        self.svm_linear = svm.SVC(kernel="precomputed", C=c)
        self.svm_linear.fit(self.kernel, y_train)

    def predict(self, x_test: np.ndarray, y_test=None):
        x_test = np.asarray(x_test, float)
        x_test_matrix = self.quantum_kernel(x_test, self.x_train, symmetric=False)
        self.predictions = self.svm_linear.predict(x_test_matrix)
        return self.predictions


if __name__ == "__main__":
    data = pd.read_csv("SUSY.csv.gz", nrows=100_000, header=None)
    data = data.dropna(subset=[TARGET, *RELEVANT_FEATURES])

    scaler = MinMaxScaler(feature_range=(-np.pi / 4, np.pi / 4))
    data_features = scaler.fit_transform(data[RELEVANT_FEATURES])
    data_target = data[TARGET].to_numpy()

    x_train = data_features[:800]
    y_train = data_target[:800]
    x_test = data_features[-1000:]
    y_test = data_target[-1000:]

    fm = z_feature_map(feature_dimension=len(RELEVANT_FEATURES), reps=2)

    shot_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1024, 2048, 4096]

    results_path = "shots-results.txt"
    with open(results_path, "w") as f:
        f.write("shots\taccuracy\tprecision\trecall\tf1\n")

        for shots in shot_list:
            qc = QCalculator(
                fm,
                shots=shots,
                optimization_level=1,
                show_progress=False,
                a_workers=65,
            )

            qc.calculate_kernel(x_train)
            qc.svm(c=1.0, y_train=y_train)
            preds = qc.predict(x_test, y_test)

            met_ac = metrics.accuracy_score(y_true=y_test, y_pred=preds)
            met_pr = metrics.precision_score(y_true=y_test, y_pred=preds)
            met_rec = metrics.recall_score(y_true=y_test, y_pred=preds)
            met_f1 = metrics.f1_score(y_true=y_test, y_pred=preds)

            f.write(
                f"{shots}\t{met_ac:.4f}\t{met_pr:.4f}\t{met_rec:.4f}\t{met_f1:.4f}\n"
            )

