from sub_qfl_client import SUB_QFL_Client
from torchquantum.dataset import MNIST
import torch
from copy import deepcopy
from Remote_model import Remoted_QFCModel
import torch.nn.functional as F
from utils import generate_client_qubits_list, split_dataset
# DQC的key components
from components import Layer, Topology
import circuit_remapper as circuit_remapper
from virtual_model import VirtualGlobalModel

class SUB_QFL_Server():
    def __init__(self, num_client,classes, lr, epochs, dataset_name, dev, local_epochs):
        self.num_client = num_client
        self.lr = lr
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.dev = dev
        self.local_epochs = local_epochs
        self.use_qiskit = False
        self.weight_decay = 1e-4
        if self.dataset_name == "mnist":
            dataset = MNIST(
            root="./mnist_data",
            train_valid_split_ratio=[0.9, 0.1],
            digits_of_interest=classes,
            )

            # self.virtual_global_model = None # wait to define
            # 把QFCmodel转化为电路形式

        self.test_set = torch.utils.data.DataLoader(
            dataset["test"], batch_size=32, shuffle=False
        )
        # NOTE: wait to split dataset in NonIID way
        self.data_clients = split_dataset(dataset["train"], self.num_client)
        # IF YOU WANT TO FIX THE QUANTUM RESOURCES, PLEASE COMMENT THE FOLLOWING LINE
        # self.num_qubits = generate_client_qubits_list(self.num_client, min_qubits=2, max_qubits=4)
        self.num_qubits = [3, 3, 4]
        self.global_n_qubits = sum(self.num_qubits)

        # cid -> number of qubits
        self.usr_to_qubits = self.distribute_client_qubits()
        # Initial Clients
        # Client 使用Remoted QFC model
        # !!!!This is just for testing 
        self.usr_to_qubits =[[0,1,2],[0,1,2],[0,1,2,3]]
        print("DEBUG:USR_TO_QUBITS",self.usr_to_qubits)
        self.clients = [
            SUB_QFL_Client(cid,self.usr_to_qubits[cid],torch.utils.data.DataLoader(
            self.data_clients[cid], batch_size=32, shuffle=True
            )
            ,self.lr,Remoted_QFCModel(wires=self.usr_to_qubits[cid], n_wires=self.num_qubits[cid]),
            self.dev,
            self.local_epochs,self.num_qubits[cid])
            for cid in range(self.num_client)]

        models = []
        for c in self.clients:
            models.append(self.clients[c.cid].model)

        # build virtual global model
        self.global_model = VirtualGlobalModel(models,n_qubits=self.global_n_qubits,qubits_list=self.num_qubits)

        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def distribute_client_qubits(self):
        usr_to_qubits = dict()
        point = 0
        for cid in range(len(self.num_qubits)):
            usr_to_qubits[cid] = list(range(point, point+self.num_qubits[cid]))
            point += self.num_qubits[cid]
        return usr_to_qubits


    def aggregation(self,std_dict):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([std_dict[i][k].float() for i in range(self.num_client)], 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        for c in self.clients:
            c.model.load_state_dict(global_dict)

    def build_batches(self):
        batches = [next(iter(c.dataloader)) for c in self.clients]
        return batches

    def run(self):

        for epoch in range(self.epochs):
            # NOTE:本地电路的训练调试
            print(f"--- Local Round {epoch+1} ---")
            # # Local training without global entangle params update
            self.global_model.freeze_global_params()
            for c in self.clients:
                # 估计就是因为他的dev不一样，重新新建了一个dev了
                c.run(self.global_model)

            print(f"--- Server Round {epoch+1} ---")
            self.global_model.frozen_global_params()
            x = self.build_batches()
            # Local training with global entangle params update
            output_all,targets = self.global_model(x)
            target_all = torch.cat(targets, dim=0)
            _, indices = output_all.topk(1, dim=1)
            # print(indices)

            masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
            size = target_all.shape[0]
            corrects = masks.sum().item()
            accuracy = corrects / size
            loss = F.nll_loss(output_all, target_all)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"test set accuracy: {accuracy}")
            print(f"test set loss: {loss}")