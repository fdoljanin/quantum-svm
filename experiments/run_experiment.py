#!/usr/bin/env python3
"""
Generic experiment runner for QSVM experiments.

Usage:
    python experiments/run_experiment.py experiments.configs.my_config
    python experiments/run_experiment.py experiments.configs.shot_sweep --output results/
"""

import sys
import argparse
import importlib
from pathlib import Path

from qsvm import QSVM
from qsvm.data import DataPipeline
from qsvm.evaluation import save_results


def run_experiment(config_module: str, output_dir: str = "results"):
    """
    Run experiment from configuration module.

    Args:
        config_module: Python module path to config (e.g., "experiments.configs.my_config")
        output_dir: Directory to save results
    """
    # Import config module
    try:
        module = importlib.import_module(config_module)
    except ImportError as e:
        print(f"Error importing config module '{config_module}': {e}")
        sys.exit(1)

    # Get config object
    if not hasattr(module, 'config'):
        print(f"Config module '{config_module}' must define 'config' variable")
        sys.exit(1)

    config = module.config

    # Optional: Get custom output filename
    output_filename = getattr(module, 'output_filename', None)
    if output_filename is None:
        output_filename = f"{config.name or 'experiment'}.json"

    # Load data
    print(f"Loading data from {config.data.data_path}...")
    pipeline = DataPipeline.from_config(config.data)
    x_train, y_train, x_test, y_test = pipeline.load_and_split()

    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
    print(f"Feature range: {pipeline.get_feature_range()}")

    # Create QSVM
    print(f"Creating QSVM with {config.kernel.strategy} kernel...")
    qsvm = QSVM.from_config(config)

    # Run experiment
    print("Computing training kernel...")
    qsvm.compute_train_kernel(x_train)

    print("Training SVM...")
    qsvm.train(x_train, y_train)

    print("Computing test kernel and predicting...")
    predictions = qsvm.predict(x_test)

    print("Evaluating...")
    from qsvm.evaluation import evaluate_predictions
    metrics = evaluate_predictions(y_test, predictions)

    # Print metrics
    print("\nResults:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1_score:.4f}")

    # Save results
    from qsvm.config.types import ExperimentResult
    result = ExperimentResult(
        config=config,
        metrics=metrics,
        predictions=predictions,
        kernel_train=qsvm.kernel_matrix_train,
    )

    output_path = Path(output_dir) / output_filename
    save_results(result, output_path, include_predictions=False, include_kernel=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run QSVM experiment from configuration module"
    )
    parser.add_argument(
        "config_module",
        help="Python module path to config (e.g., experiments.configs.my_config)"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results (default: results/)"
    )

    args = parser.parse_args()

    run_experiment(args.config_module, args.output)


if __name__ == "__main__":
    main()
