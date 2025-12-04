from qfl_client import QFL_Client
from torchquantum.dataset import MNIST
import torch
from torch.utils.data import random_split
from copy import deepcopy
from qfc_model import QFCModel
import torch.nn.functional as F

class QFL_Server():
    def __init__(self, num_client, lr, epochs, dataset_name, dev, local_epochs):
        self.num_client = num_client
        self.lr = lr
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.dev = dev
        self.local_epochs = local_epochs
        if self.dataset_name == "mnist":
            dataset = MNIST(
            root="./mnist_data",
            train_valid_split_ratio=[0.9, 0.1],
            digits_of_interest=[3, 6],
            )
            self.global_model = QFCModel()
        self.test_set = torch.utils.data.DataLoader(
            dataset["test"], batch_size=32, shuffle=False
        )
        self.data_clients = self.split_dataset(dataset["train"])
        self.clients = [
            QFL_Client(cid,torch.utils.data.DataLoader(
            self.data_clients[cid], batch_size=32, shuffle=True
            ),self.lr,deepcopy(self.global_model),self.dev,self.local_epochs)
            for cid in range(self.num_client)]

    def split_dataset(self, dataset):
        total_size = len(dataset)
        # 整除部分，每个 client 至少分到的数量
        split_size = total_size // self.num_client
        # 余数部分，前 remainder 个 client 会多拿 1 个数据
        remainder = total_size % self.num_client

        # 生成一个列表，包含每个 client 应得的数据长度
        # 例如：总量10，分3份 -> lengths=[4, 3, 3]
        lengths = [split_size + 1 if i < remainder else split_size for i in range(self.num_client)]

        # 3. 执行拆分
        # generator参数可选，用于固定随机种子，保证每次跑分法一致
        client_datasets = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))

        # --- 验证结果 ---
        print(f"原始数据集大小: {total_size}")
        print(f"客户端数量: {self.num_client}")
        print(f"拆分方案 (lengths): {lengths}")

        # 打印前几个 client 的数据集大小
        for i, client_ds in enumerate(client_datasets):
            print(f"Client {i} 数据量: {len(client_ds)}")
        return client_datasets

    def aggregation(self,std_dict):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([std_dict[i][k].float() for i in range(self.num_client)], 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        for c in self.clients:
            c.model.load_state_dict(global_dict)

    def test(self,qiskit=False):
        target_all = []
        output_all = []
        with torch.no_grad():
            for feed_dict in self.test_set:
                inputs = feed_dict["image"].to(self.dev)
                targets = feed_dict["digit"].to(self.dev)

                outputs = self.global_model(inputs, use_qiskit=qiskit)

                target_all.append(targets)
                output_all.append(outputs)
            target_all = torch.cat(target_all, dim=0)
            output_all = torch.cat(output_all, dim=0)

        _, indices = output_all.topk(1, dim=1)
        masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
        size = target_all.shape[0]
        corrects = masks.sum().item()
        accuracy = corrects / size
        loss = F.nll_loss(output_all, target_all).item()

        print(f"test set accuracy: {accuracy}")
        print(f"test set loss: {loss}")
        pass

    def run(self):
        std_dict = {}
        for epoch in range(self.epochs):
            print(f"--- Server Round {epoch+1} ---")
            for c in self.clients:
                std_dict[c.cid] = c.run()
            # 聚合模型参数（简单平均）
            self.aggregation(std_dict)
            self.test()