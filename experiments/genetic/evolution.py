import json
import time
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import numpy as np

from qsvm.feature_maps.custom_ansatz import CustomAnsatz, random_ansatz
from experiments.genetic.operators import create_offspring
from qsvm.config.types import ExperimentConfig


def _evaluate_individual_wrapper(args):
    """
    Wrapper function for parallel fitness evaluation.
    Must be at module level to be picklable.
    """
    individual, fitness_function, index = args
    return index, fitness_function(individual)


@dataclass
class GAConfig:
    population_size: int = 20
    max_generations: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    elite_count: int = 2
    early_stopping_generations: int = 10
    feature_dimension: int = 8
    min_circuit_depth: int = 15
    max_circuit_depth: int = 150
    entanglement_prob: float = 0.3


@dataclass
class GenerationResult:
    generation: int
    population: List[CustomAnsatz]
    fitness_values: List[float]
    best_individual: CustomAnsatz
    best_fitness: float
    mean_fitness: float
    evaluation_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation': self.generation,
            'population': [
                {
                    'genome': ind.to_genome_string(),
                    'fitness': float(fitness)
                }
                for ind, fitness in zip(self.population, self.fitness_values)
            ],
            'best_individual': {
                'genome': self.best_individual.to_genome_string(),
                'fitness': float(self.best_fitness)
            },
            'best_fitness': float(self.best_fitness),
            'mean_fitness': float(self.mean_fitness),
            'evaluation_time_seconds': float(self.evaluation_time)
        }


class GeneticAlgorithm:
    def __init__(
        self,
        ga_config: GAConfig,
        fitness_function: Callable[[CustomAnsatz], float],
        output_path: Optional[Path] = None
    ):
        self.config = ga_config
        self.fitness_function = fitness_function
        self.output_path = output_path

        self.population: List[CustomAnsatz] = []
        self.fitness_values: List[float] = []

        self.generation_results: List[GenerationResult] = []
        self.best_overall: Optional[CustomAnsatz] = None
        self.best_overall_fitness: float = -1.0
        self.best_overall_generation: int = -1

        self.generations_without_improvement: int = 0

    def initialize_population(self) -> None:
        print(f"Initializing population of {self.config.population_size}...")
        self.population = [
            random_ansatz(
                feature_dimension=self.config.feature_dimension,
                min_depth=self.config.min_circuit_depth,
                max_depth=self.config.max_circuit_depth,
                entanglement_prob=self.config.entanglement_prob
            )
            for _ in range(self.config.population_size)
        ]

    def evaluate_population(self) -> None:
        """
        Evaluate each individual in the population using parallel processes.
        Each individual evaluation runs in its own process.
        """
        print(f"  Evaluating population of {len(self.population)} in parallel processes...")

        # Prepare arguments for parallel evaluation
        eval_args = [
            (individual, self.fitness_function, i)
            for i, individual in enumerate(self.population)
        ]

        # Use population_size processes, one for each individual
        num_processes = len(self.population)

        with Pool(processes=num_processes) as pool:
            results = pool.map(_evaluate_individual_wrapper, eval_args)

        # Sort results by index and extract fitness values
        results.sort(key=lambda x: x[0])
        self.fitness_values = [fitness for _, fitness in results]

        print("  ✓ Evaluation complete")

    def evolve_generation(self, generation: int) -> GenerationResult:
        start_time = time.time()

        print(f"\n=== Generation {generation} ===")

        self.evaluate_population()

        best_idx = np.argmax(self.fitness_values)
        best_fitness = self.fitness_values[best_idx]
        best_individual = self.population[best_idx]
        mean_fitness = np.mean(self.fitness_values)

        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Mean fitness: {mean_fitness:.4f}")
        print(f"Best genome: {best_individual.to_genome_string()}")

        if best_fitness > self.best_overall_fitness:
            self.best_overall_fitness = best_fitness
            self.best_overall = best_individual.copy()
            self.best_overall_generation = generation
            self.generations_without_improvement = 0
            print("*** New best overall! ***")
        else:
            self.generations_without_improvement += 1

        evaluation_time = time.time() - start_time
        gen_result = GenerationResult(
            generation=generation,
            population=[ind.copy() for ind in self.population],
            fitness_values=self.fitness_values.copy(),
            best_individual=best_individual.copy(),
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            evaluation_time=evaluation_time
        )
        self.generation_results.append(gen_result)

        if generation < self.config.max_generations - 1:
            sorted_indices = np.argsort(self.fitness_values)[::-1]

            elite = [self.population[i].copy() for i in sorted_indices[:self.config.elite_count]]

            offspring_size = self.config.population_size - self.config.elite_count
            offspring = create_offspring(
                population=self.population,
                fitness_values=self.fitness_values,
                offspring_size=offspring_size,
                crossover_rate=self.config.crossover_rate,
                mutation_rate=self.config.mutation_rate,
                tournament_size=self.config.tournament_size,
                feature_dimension=self.config.feature_dimension
            )

            self.population = elite + offspring

        return gen_result

    def run(self) -> None:
        print("=" * 60)
        print("GENETIC ALGORITHM - QUANTUM FEATURE MAP EVOLUTION")
        print("=" * 60)

        self.initialize_population()

        for generation in range(self.config.max_generations):
            self.evolve_generation(generation)

            # Save results after each generation if output_path is provided
            if self.output_path is not None:
                self.save_results(self.output_path)

            if self.generations_without_improvement >= self.config.early_stopping_generations:
                print(f"\nEarly stopping: No improvement for {self.config.early_stopping_generations} generations")
                break

        print("\n" + "=" * 60)
        print("EVOLUTION COMPLETE")
        print("=" * 60)
        print(f"Best fitness: {self.best_overall_fitness:.4f}")
        print(f"Best generation: {self.best_overall_generation}")
        print(f"Best genome: {self.best_overall.to_genome_string()}")
        print("=" * 60)

    def save_results(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        best_fitness_per_gen = [gr.best_fitness for gr in self.generation_results]
        mean_fitness_per_gen = [gr.mean_fitness for gr in self.generation_results]

        results = {
            'ga_config': {
                'population_size': self.config.population_size,
                'max_generations': self.config.max_generations,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'tournament_size': self.config.tournament_size,
                'elite_count': self.config.elite_count,
                'early_stopping_generations': self.config.early_stopping_generations,
                'feature_dimension': self.config.feature_dimension,
                'min_circuit_depth': self.config.min_circuit_depth,
                'max_circuit_depth': self.config.max_circuit_depth,
                'entanglement_prob': self.config.entanglement_prob
            },
            'generations': [gr.to_dict() for gr in self.generation_results],
            'convergence': {
                'best_fitness_per_generation': best_fitness_per_gen,
                'mean_fitness_per_generation': mean_fitness_per_gen,
                'overall_best': {
                    'genome': self.best_overall.to_genome_string(),
                    'fitness': float(self.best_overall_fitness),
                    'generation': self.best_overall_generation
                }
            },
            'total_generations': len(self.generation_results),
            'stopped_early': self.generations_without_improvement >= self.config.early_stopping_generations
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")
