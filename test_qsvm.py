#!/usr/bin/env python3
"""Quick test to verify QSVM installation and basic functionality."""

from qsvm import QSVM
from qsvm.config import default_statevector_config
from qsvm.data import DataPipeline
from dataclasses import replace

# Create a small test config (reduced data for quick test)
test_config = replace(
    default_statevector_config,
    name="quick_test",
    data=replace(
        default_statevector_config.data,
        train_size=500,  # Small size for quick test
        test_size=500,
        scale_factor=0.5
    )
)

print("Loading data...")
pipeline = DataPipeline.from_config(test_config.data)
x_train, y_train, x_test, y_test = pipeline.load_and_split()

print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
print(f"Feature range: {pipeline.get_feature_range()}")

print("\nCreating QSVM...")
qsvm = QSVM.from_config(test_config)

print("Running experiment (this will take a few minutes)...")
result = qsvm.fit_evaluate(x_train, y_train, x_test, y_test)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Accuracy:  {result.metrics.accuracy:.4f}")
print(f"Precision: {result.metrics.precision:.4f}")
print(f"Recall:    {result.metrics.recall:.4f}")
print(f"F1 Score:  {result.metrics.f1_score:.4f}")
print("="*50)
print("\nTest successful! QSVM is working correctly.")
