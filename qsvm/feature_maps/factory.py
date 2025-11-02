from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qsvm.config.types import FeatureMapConfig

def create_feature_map(config: FeatureMapConfig):
    """
    Create Qiskit feature map from configuration.

    Args:
        config: FeatureMapConfig specifying feature map type and parameters

    Returns:
        Qiskit feature map circuit

    Raises:
        ValueError: If feature map type is not recognized
    """
    if config.type == "z":
        return ZFeatureMap(
            feature_dimension=config.feature_dimension,
            reps=config.reps,
        )
    elif config.type == "zz":
        return ZZFeatureMap(
            feature_dimension=config.feature_dimension,
            reps=config.reps,
        )
    elif config.type == "pauli":
        paulis = config.paulis or ['X', 'Y', 'Z']
        return PauliFeatureMap(
            feature_dimension=config.feature_dimension,
            reps=config.reps,
            paulis=paulis,
        )
    else:
        raise ValueError(f"Unknown feature map type: {config.type}")
