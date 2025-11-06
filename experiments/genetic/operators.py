import random
from typing import List, Tuple
from qsvm.feature_maps.custom_ansatz import CustomAnsatz, random_gate, Gate


def tournament_selection(
    population: List[CustomAnsatz],
    fitness_values: List[float],
    tournament_size: int = 3
) -> CustomAnsatz:
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_idx].copy()


def two_point_crossover(
    parent1: CustomAnsatz,
    parent2: CustomAnsatz
) -> Tuple[CustomAnsatz, CustomAnsatz]:
    genome1 = parent1.genome
    genome2 = parent2.genome

    if len(genome1) < 2 or len(genome2) < 2:
        return parent1.copy(), parent2.copy()

    min_len = min(len(genome1), len(genome2))
    if min_len < 3:
        mid1 = len(genome1) // 2
        mid2 = len(genome2) // 2
        child1_genome = genome1[:mid1] + genome2[mid2:]
        child2_genome = genome2[:mid2] + genome1[mid1:]
    else:
        point1 = random.randint(1, min_len - 2)
        point2 = random.randint(point1 + 1, min_len - 1)
        child1_genome = genome1[:point1] + genome2[point1:point2] + genome1[point2:]
        child2_genome = genome2[:point1] + genome1[point1:point2] + genome2[point2:]

    child1 = CustomAnsatz(child1_genome, parent1.feature_dimension)
    child2 = CustomAnsatz(child2_genome, parent2.feature_dimension)

    return child1, child2


def mutate(
    ansatz: CustomAnsatz,
    mutation_rate: float = 0.3,
    feature_dimension: int = 8
) -> CustomAnsatz:
    genome = [g for g in ansatz.genome]

    if random.random() > mutation_rate:
        return CustomAnsatz(genome, feature_dimension)

    num_mutations = random.randint(1, 3)

    for _ in range(num_mutations):
        if len(genome) == 0:
            genome.append(random_gate(feature_dimension))
            continue

        mutation_type = random.choice([
            'gate_type', 'qubit', 'parameter', 'add', 'delete'
        ])

        if mutation_type == 'gate_type' and len(genome) > 0:
            idx = random.randint(0, len(genome) - 1)
            gate_type, qubits, param_idx = genome[idx]

            if len(qubits) == 1:
                if param_idx is not None:
                    new_gate_type = random.choice(['rx', 'ry', 'rz'])
                else:
                    new_gate_type = random.choice(['h', 'x', 'y', 'z'])
            else:
                if param_idx is not None:
                    new_gate_type = random.choice(['crx', 'cry', 'crz'])
                else:
                    new_gate_type = random.choice(['cx', 'cz', 'cy'])

            genome[idx] = (new_gate_type, qubits, param_idx)

        elif mutation_type == 'qubit' and len(genome) > 0:
            idx = random.randint(0, len(genome) - 1)
            gate_type, qubits, param_idx = genome[idx]

            if len(qubits) == 1:
                new_qubits = [random.randint(0, feature_dimension - 1)]
            else:
                new_qubits = random.sample(range(feature_dimension), 2)

            genome[idx] = (gate_type, new_qubits, param_idx)

        elif mutation_type == 'parameter' and len(genome) > 0:
            idx = random.randint(0, len(genome) - 1)
            gate_type, qubits, param_idx = genome[idx]

            if param_idx is not None:
                new_param_idx = random.randint(0, feature_dimension - 1)
                genome[idx] = (gate_type, qubits, new_param_idx)

        elif mutation_type == 'add':
            idx = random.randint(0, len(genome))
            genome.insert(idx, random_gate(feature_dimension))

        elif mutation_type == 'delete' and len(genome) > 5:
            idx = random.randint(0, len(genome) - 1)
            genome.pop(idx)

    return CustomAnsatz(genome, feature_dimension)


def create_offspring(
    population: List[CustomAnsatz],
    fitness_values: List[float],
    offspring_size: int,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.3,
    tournament_size: int = 3,
    feature_dimension: int = 8
) -> List[CustomAnsatz]:
    offspring = []

    while len(offspring) < offspring_size:
        parent1 = tournament_selection(population, fitness_values, tournament_size)
        parent2 = tournament_selection(population, fitness_values, tournament_size)

        if random.random() < crossover_rate:
            child1, child2 = two_point_crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        child1 = mutate(child1, mutation_rate, feature_dimension)
        child2 = mutate(child2, mutation_rate, feature_dimension)

        offspring.append(child1)
        if len(offspring) < offspring_size:
            offspring.append(child2)

    return offspring[:offspring_size]
