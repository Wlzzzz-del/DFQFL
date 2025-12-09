from sub_qfl_client import SUB_QFL_Client
from tqdm import tqdm
from torchquantum.dataset import MNIST
import torch
from copy import deepcopy
from Remote_model import Remoted_QFCModel
import torch.nn.functional as F
from utils import generate_client_qubits_list
# DQC的key components
from components import Layer, Topology
import circuit_remapper as circuit_remapper
from virtual_model import VirtualGlobalModel
from process_data import generate_dataset, split_data_federated, generate_client_percentage_config
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class SUB_QFL_Server():
    def __init__(self, num_client,classes, lr, epochs, dataset_name, dev, local_epochs, weight_decay=1e-4, batch_size=32,min_qubits=2, max_qubits=4):
        self.num_client = num_client
        self.lr = lr
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.dev = dev
        self.local_epochs = local_epochs
        self.use_qiskit = False
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.min_qubits = min_qubits
        self.max_qubits = max_qubits

        if self.dataset_name == "mnist":
            X, y = generate_dataset(self.dataset_name,classes,n_samples=2000)
            clients_config_arg=generate_client_percentage_config(self.num_client)
            self.data_clients, (X_test, y_test) = split_data_federated(X, y, clients_config_arg, test_frac=0.2, val_frac=0.1, random_state=42)

        elif self.dataset_name == "cifar10":
            pass

        self.test_set = torch.utils.data.DataLoader(
           TensorDataset(X_test,y_test), batch_size=len(X_test), shuffle=False
        )

        # Heterogeneous Qubits Distribution of clients nisq devices
        self.num_qubits = generate_client_qubits_list(self.num_client, min_qubits=min_qubits, max_qubits=max_qubits)
        print("LOG: Generated client qubits list:", self.num_qubits)

        self.global_n_qubits = sum(self.num_qubits)
        print("LOG: Global number of qubits:", self.global_n_qubits)

        self.usr_to_qubits = self.distribute_globalqbit_to_client()
        self.usr_to_qubits = self.gen_client_qubits_list()
        # print("DEBUG:USR_TO_QUBITS",self.usr_to_qubits)

        self.clients = [
            SUB_QFL_Client(cid,self.usr_to_qubits[cid],
            # Ctype->cid->train_data->x,y
            torch.utils.data.DataLoader(TensorDataset(self.data_clients[cid][0][0][0],self.data_clients[cid][0][0][1]), batch_size=self.batch_size, shuffle=True)
            ,self.lr,Remoted_QFCModel(wires=self.usr_to_qubits[cid], n_wires=self.num_qubits[cid]),
            self.dev,
            self.local_epochs,self.num_qubits[cid],self.weight_decay)
            for cid in range(self.num_client)]

        models = []
        for c in self.clients:
            models.append(self.clients[c.cid].model)

        # build virtual global model
        self.global_model = VirtualGlobalModel(models,n_qubits=self.global_n_qubits,qubits_list=self.num_qubits)
        # print("LOG: Global Model Parameter:", self.global_model.named_parameters())
        for name, param in self.global_model.named_parameters():
            if param.requires_grad:
                print(f"{name:<20} {str(param.shape):<20} {param.requires_grad}")

        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.history = {'loss': [], 'accuracy': []}

    def distribute_globalqbit_to_client(self):
        """
        Distribute global qubits to clients.
        Example: 3 clients with [3,3,4] qubits respectively
        client 0 -> [0,1,2]
        client 1 -> [3,4,5]
        client 2 -> [6,7,8,9]

        Returns:
            _dict: cid -> qubits list
        """
        usr_to_qubits = dict()
        point = 0
        for cid in range(len(self.num_qubits)):
            usr_to_qubits[cid] = list(range(point, point+self.num_qubits[cid]))
            point += self.num_qubits[cid]
        return usr_to_qubits

    def gen_client_qubits_list(self):
        """
        Set local qubits to clients.
        Example: 3 clients with [3,3,4] qubits respectively
        client 0 -> [0,1,2]
        client 1 -> [0,1,2]
        client 2 -> [0,1,2,3]

        Returns:
            _list: -> qubits list [[0,1,2], [0,1,2], [0,1,2,3]]
        """
        qubits_list = []
        for cid in range(len(self.num_qubits)):
            qubits_list.append(list(range(0,self.num_qubits[cid])))
        return qubits_list

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

    def build_test_batches(self):
        batches = [next(iter(self.test_set)) for _ in self.clients]
        # batches = [next(iter(self.test_set))]
        return batches

    def test(self):
        x = self.build_batches()
        # Local training with global entangle params update
        output_all,targets = self.global_model(x)
        target_all = torch.cat(targets, dim=0)
        _, indices = output_all.topk(1, dim=1)

        masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
        size = target_all.shape[0]
        corrects = masks.sum().item()
        accuracy = corrects / size
        # print("--- Test Accuracy: {:.4f} ---".format(accuracy))
        return accuracy

    def run(self):
        pbar = tqdm(range(self.epochs), desc="training rounds")
        for _epoch in pbar:
            # =====Local training without global entangle params update=====
            self.global_model.freeze_global_params()# freeze global entangle params donot update
            for c in self.clients:
                c.run(self.global_model)

            self.global_model.frozen_global_params()# unfreeze global entangle params to update
            x = self.build_batches()
            # Local training with global entangle params update
            output_all,targets = self.global_model(x)
            # _, indices = output_all.topk(1, dim=1)

            # masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
            # size = target_all.shape[0]
            # corrects = masks.sum().item()
            # accuracy = corrects / size
            target_all = torch.cat(targets, dim=0)
            num_classes = output_all.size(1)
            target_one_hot = F.one_hot(target_all, num_classes=num_classes).float()
            output_probs = F.softmax(output_all, dim=1)
            loss = F.mse_loss(output_probs, target_one_hot)
            # loss = F.nll_loss(output_all, target_all)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accuracy = self.test()
            self.history['loss'].append(loss.item())
            self.history['accuracy'].append(accuracy)
            pbar.set_postfix({'loss': loss.item(), 'accuracy': accuracy})


        # ===== Training complete, plot the results =====
        rounds = list(range(1, self.epochs + 1))
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(rounds, self.history['loss'], label='Server Loss', color='red', marker='o', linestyle='-')
        plt.title('Server Training Loss over Rounds')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss (NLL)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # 2. 绘制 Accuracy 曲线 
        plt.subplot(1, 2, 2)
        plt.plot(rounds, self.history['accuracy'], label='Server Accuracy', color='blue', marker='o', linestyle='-')
        plt.title('Server Test Accuracy over Rounds')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.show() # 显示图表

        print("\n--- Training complete. Plots displayed. ---")