import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import numpy as np

import torchquantum as tq
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)

from torchquantum.dataset import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from qfl_server import QFL_Server
from sub_qfl_server import SUB_QFL_Server

def main():
    # Hyperparameters
    lr = 0.0001
    epochs = 1000
    dataset = "mnist"
    classes = [0,1]
    num_client = 3
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 1
    server = "SUB_QFL" # QFL// SUB_QFL

    print("Starting DFQFL Server...")
    # Define and run DFQFL server
    if server == "QFL":
        QFL = QFL_Server(num_client, lr, epochs, dataset,dev, local_epochs)
        QFL.run()

    elif server == "SUB_QFL":
        SUB_QFL = SUB_QFL_Server(num_client,classes, lr, epochs, dataset,dev, local_epochs)
        SUB_QFL.run()
        # SUB_QFL.test()

main()