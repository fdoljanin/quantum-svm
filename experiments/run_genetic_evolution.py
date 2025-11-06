import argparse
from pathlib import Path
from dataclasses import replace
from datetime import datetime

from qsvm.data import DataPipeline
from qsvm.config.defaults import default_statevector_config
from qsvm.config.types import DataConfig, KernelConfig
from experiments.genetic import (
    GAConfig,
    GeneticAlgorithm,
    evaluate_ansatz_fitness,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run genetic algorithm for quantum feature map evolution"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/ga_evolution.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--population',
        type=int,
        default=20,
        help='Population size'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='Maximum number of generations'
    )
    parser.add_argument(
        '--mutation-rate',
        type=float,
        default=0.3,
        help='Mutation rate (0.0-1.0)'
    )
    parser.add_argument(
        '--crossover-rate',
        type=float,
        default=0.7,
        help='Crossover rate (0.0-1.0)'
    )
    parser.add_argument(
        '--train-size',
        type=int,
        default=200,
        help='Training set size'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=200,
        help='Test set size'
    )
    parser.add_argument(
        '--feature-dimension',
        type=int,
        default=8,
        help='Number of qubits (feature dimension)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=72,
        help='Number of parallel workers for kernel computation'
    )

    args = parser.parse_args()

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output)
    output_filename = f"{output_path.stem}_{datetime_str}{output_path.suffix}"
    output_path = output_path.parent / output_filename

    print("=" * 60)
    print("GENETIC ALGORITHM - QUANTUM FEATURE MAP EVOLUTION")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Mutation rate: {args.mutation_rate}")
    print(f"Crossover rate: {args.crossover_rate}")
    print(f"Training samples: {args.train_size}")
    print(f"Test samples: {args.test_size}")
    print(f"Feature dimension: {args.feature_dimension}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    data_config = DataConfig(
        data_path="raw/SUSY.csv.gz",
        nrows=10_000,
        feature_columns=(1, 2, 3, 4, 5, 6, 7, 8),
        target_column=0,
        scale_range=(-3.14159 / 2, 3.14159 / 2),
        scale_factor=0.5,
        train_size=args.train_size,
        test_size=args.test_size
    )
    pipeline = DataPipeline.from_config(data_config)
    x_train, y_train, x_test, y_test = pipeline.load_and_split()

    print(f"\nData loaded:")
    print(f"  Training: {x_train.shape}")
    print(f"  Test: {x_test.shape}")

    kernel_config = KernelConfig(
        strategy="statevector",
        cache_statevectors=True,
        show_progress=False,
        workers=args.workers
    )

    exp_config = replace(
        default_statevector_config,
        data=data_config,
        kernel=kernel_config
    )

    def fitness_function(ansatz):
        return evaluate_ansatz_fitness(
            ansatz=ansatz,
            config=exp_config,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    ga_config = GAConfig(
        population_size=args.population,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        feature_dimension=args.feature_dimension
    )

    ga = GeneticAlgorithm(ga_config, fitness_function, output_path)
    ga.run()

    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
