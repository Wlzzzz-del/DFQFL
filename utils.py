import torch
import torch.nn.functional as F
import random
import torchquantum as tq
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)
from components import Layer, Topology
import circuit_remapper as circuit_remapper
from typing import List
from torch.utils.data import random_split
import numpy as np
from sklearn.decomposition import PCA

def qiskit_compose_circs(fixed_layers: List[tq.QuantumCircuit]) -> tq.QuantumCircuit:
    """Compose subcircuits into a full circuit.

    Args:
        fixed_layers (_type_): _list of subcircuits

    Returns:
        _type_: _full circuit(Qiskit QuantumCircuit type)
    """
    circs_all = fixed_layers[0].copy_empty_like()
    for fixed_layer in fixed_layers:
        circs_all.compose(fixed_layer,qubits=range(fixed_layer.num_qubits),inplace=True)
        circs_all.append(fixed_layer)
    return circs_all

def generate_client_qubits_list(num_client: int, min_qubits: int, max_qubits: int) -> List[int]:
    """
    Generate the random qubit number of clients' subcircuits.

    Parameters:
    num_client (int): Number of clients.
    min_qubits (int): Minimum number of qubits for each client's subcircuit.
    max_qubits (int): Maximum number of qubits for each client's subcircuit.
    Returns:
    List[int]: each element is qubits number of each client's subcircuit.
    """
    random_list = []

    if num_client <= 0:
        return random_list

    for _ in range(num_client):
        random_int = random.randint(min_qubits, max_qubits)
        # random_list.append(random_int)

        # For testing purpose, we fix the qubits number to be 2,3,4
        # random_list.append(2)# 2的时候能到6、70
        # 2\3 是效果最好的
    random_list=[2,3]

    return random_list

def expanded_imgs_for_wires(batch_data,num_client):
    """expand img data to fit the qubits number of qdev dimension which is equal to all clients

    Args:
        batch_data (_type_): _input img data
        num_client (_type_): _number of clients

    Returns:
        _type_: _expanded img data
    """
    to_build = [batch_data]
    for _ in range(num_client - 1):
        to_build.append(torch.zeros_like(batch_data))
    return torch.cat(to_build, dim=0)