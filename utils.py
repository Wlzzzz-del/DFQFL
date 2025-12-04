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
    """
    random_list = []

    if num_client <= 0:
        return random_list

    for _ in range(num_client):
        random_int = random.randint(min_qubits, max_qubits)
        random_list.append(random_int)

    return random_list

def split_dataset(dataset, num_client):
    total_size = len(dataset)
    # 整除部分，每个 client 至少分到的数量
    split_size = total_size // num_client
    # 余数部分，前 remainder 个 client 会多拿 1 个数据
    remainder = total_size % num_client

    # 生成一个列表，包含每个 client 应得的数据长度
    # 例如：总量10，分3份 -> lengths=[4, 3, 3]
    lengths = [split_size + 1 if i < remainder else split_size for i in range(num_client)]

    # 3. 执行拆分
    # generator参数可选，用于固定随机种子，保证每次跑分法一致
    client_datasets = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))

    # --- 验证结果 ---
    print(f"原始数据集大小: {total_size}")
    print(f"客户端数量: {num_client}")
    print(f"拆分方案 (lengths): {lengths}")

    # 打印前几个 client 的数据集大小
    for i, client_ds in enumerate(client_datasets):
        print(f"Client {i} 数据量: {len(client_ds)}")
    return client_datasets