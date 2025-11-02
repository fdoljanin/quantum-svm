#!/usr/bin/env python3
"""
Batch experiment runner for parameter sweeps.

Usage:
    python experiments/run_batch.py experiments.configs.shot_sweep --output results/shot_sweep/
"""

import sys
import argparse
import importlib
from pathlib import Path

from qsvm import QSVM
from qsvm.data import DataPipeline
from qsvm.evaluation import ResultsWriter


def run_batch_experiments(config_module: str, output_dir: str = "results"):
    """
    Run batch experiments from configuration module.

    Args:
        config_module: Python module path to config
        output_dir: Directory to save results
    """
    # Import config module
    try:
        module = importlib.import_module(config_module)
    except ImportError as e:
        print(f"Error importing config module '{config_module}': {e}")
        sys.exit(1)

    # Get batch configs
    if not hasattr(module, 'get_configs'):
        print(f"Config module '{config_module}' must define 'get_configs()' function")
        sys.exit(1)

    configs = module.get_configs()
    print(f"Running {len(configs)} experiments...")

    # Create results writer
    output_path = Path(output_dir) / "batch_results.json"
    writer = ResultsWriter(output_path)

    # Run each experiment
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running: {config.name}")

        # Load data
        pipeline = DataPipeline.from_config(config.data)
        x_train, y_train, x_test, y_test = pipeline.load_and_split()

        # Run experiment
        qsvm = QSVM.from_config(config)
        result = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)

        # Print metrics
        print(f"  Accuracy: {result.metrics.accuracy:.4f}, F1: {result.metrics.f1_score:.4f}")

        # Add to results
        writer.add_result(result)

    # Save all results
    writer.save()
    print(f"\nAll results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch QSVM experiments from configuration module"
    )
    parser.add_argument(
        "config_module",
        help="Python module path to config"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    run_batch_experiments(args.config_module, args.output)


if __name__ == "__main__":
    main()
