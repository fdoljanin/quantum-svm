from typing import List, Tuple, Optional, Union
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

Gate = Tuple[str, List[int], Optional[int]]

class CustomAnsatz:
    def __init__(self, genome: List[Gate], feature_dimension: int):
        self.genome = genome
        self.feature_dimension = feature_dimension
        self._circuit = None
        self._parameters = None

    @property
    def num_qubits(self) -> int:
        return self.feature_dimension

    def _build_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.feature_dimension)

        if self._parameters is None:
            self._parameters = [
                Parameter(f'x[{i}]')
                for i in range(self.feature_dimension)
            ]

        for gate_type, qubits, param_idx in self.genome:
            if gate_type == 'rx':
                circuit.rx(self._parameters[param_idx], qubits[0])
            elif gate_type == 'ry':
                circuit.ry(self._parameters[param_idx], qubits[0])
            elif gate_type == 'rz':
                circuit.rz(self._parameters[param_idx], qubits[0])
            elif gate_type == 'h':
                circuit.h(qubits[0])
            elif gate_type == 'x':
                circuit.x(qubits[0])
            elif gate_type == 'y':
                circuit.y(qubits[0])
            elif gate_type == 'z':
                circuit.z(qubits[0])
            elif gate_type == 's':
                circuit.s(qubits[0])
            elif gate_type == 't':
                circuit.t(qubits[0])
            elif gate_type == 'cx':
                circuit.cx(qubits[0], qubits[1])
            elif gate_type == 'cz':
                circuit.cz(qubits[0], qubits[1])
            elif gate_type == 'cy':
                circuit.cy(qubits[0], qubits[1])
            elif gate_type == 'swap':
                circuit.swap(qubits[0], qubits[1])
            elif gate_type == 'crx':
                circuit.crx(self._parameters[param_idx], qubits[0], qubits[1])
            elif gate_type == 'cry':
                circuit.cry(self._parameters[param_idx], qubits[0], qubits[1])
            elif gate_type == 'crz':
                circuit.crz(self._parameters[param_idx], qubits[0], qubits[1])
            else:
                raise ValueError(f"Unknown gate type: {gate_type}")

        return circuit

    def assign_parameters(self, values: Union[List, np.ndarray]) -> QuantumCircuit:
        if self._circuit is None:
            self._circuit = self._build_circuit()

        param_dict = {}
        for param in self._circuit.parameters:
            param_name = param.name
            idx = int(param_name.split('[')[1].split(']')[0])
            param_dict[param] = float(values[idx])

        return self._circuit.assign_parameters(param_dict)

    def inverse(self) -> 'CustomAnsatz':
        inverted_genome = []
        for gate_type, qubits, param_idx in reversed(self.genome):
            inverted_genome.append((gate_type, qubits, param_idx))

        return CustomAnsatz(inverted_genome, self.feature_dimension)

    def to_genome_string(self) -> str:
        parts = []
        for gate_type, qubits, param_idx in self.genome:
            gate_str = gate_type.upper()
            qubit_str = ''.join(map(str, qubits))
            if param_idx is not None:
                parts.append(f"{gate_str}{qubit_str}(p{param_idx})")
            else:
                parts.append(f"{gate_str}{qubit_str}")
        return '-'.join(parts)

    @classmethod
    def from_genome_string(cls, genome_str: str, feature_dimension: int) -> 'CustomAnsatz':
        genome = []
        for gate_str in genome_str.split('-'):
            gate_type = ''
            i = 0
            while i < len(gate_str) and gate_str[i].isalpha():
                gate_type += gate_str[i].lower()
                i += 1

            qubits = []
            while i < len(gate_str) and (gate_str[i].isdigit()):
                qubits.append(int(gate_str[i]))
                i += 1

            param_idx = None
            if i < len(gate_str) and gate_str[i] == '(':
                param_str = gate_str[i+1:gate_str.index(')')]
                param_idx = int(param_str[1:])

            genome.append((gate_type, qubits, param_idx))

        return cls(genome, feature_dimension)

    def copy(self) -> 'CustomAnsatz':
        return CustomAnsatz([g for g in self.genome], self.feature_dimension)


def random_gate(feature_dimension: int, entanglement_prob: float = 0.3) -> Gate:
    single_qubit_parametric = ['rx', 'ry', 'rz']
    single_qubit_fixed = ['h', 'x', 'y', 'z']
    two_qubit_parametric = ['crx', 'cry', 'crz']
    two_qubit_fixed = ['cx', 'cz', 'cy']

    if random.random() < entanglement_prob:
        q1, q2 = random.sample(range(feature_dimension), 2)
        if random.random() < 0.5:
            gate_type = random.choice(two_qubit_parametric)
            param_idx = random.randint(0, feature_dimension - 1)
            return (gate_type, [q1, q2], param_idx)
        else:
            gate_type = random.choice(two_qubit_fixed)
            return (gate_type, [q1, q2], None)
    else:
        q = random.randint(0, feature_dimension - 1)
        if random.random() < 0.7:
            gate_type = random.choice(single_qubit_parametric)
            param_idx = random.randint(0, feature_dimension - 1)
            return (gate_type, [q], param_idx)
        else:
            gate_type = random.choice(single_qubit_fixed)
            return (gate_type, [q], None)


def random_ansatz(feature_dimension: int,
                  min_depth: int = 5,
                  max_depth: int = 15,
                  entanglement_prob: float = 0.3) -> CustomAnsatz:
    depth = random.randint(min_depth, max_depth)
    genome = [
        random_gate(feature_dimension, entanglement_prob)
        for _ in range(depth)
    ]
    return CustomAnsatz(genome, feature_dimension)
