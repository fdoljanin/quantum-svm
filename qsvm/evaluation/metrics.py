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
) -> None:
    """
    Save experiment results to JSON file.

    Args:
        result: ExperimentResult to save
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = result.to_dict()

    result_dict['timestamp'] = datetime.now().isoformat()
    result_dict['qsvm_version'] = "2.0.0"

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
