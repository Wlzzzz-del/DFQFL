import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torch

class SUB_QFL_Client():
    def __init__(self,cid,qbits_idx,dataloader,lr,model,dev,epochs,num_qubits):
        self.cid = cid
        self.qbits_idx = qbits_idx
        self.dataloader = dataloader
        self.lr = lr
        self.model = model
        self.dev = dev
        self.local_epochs = epochs
        self.weight_decay = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.local_epochs)
        self.num_qubits = num_qubits

    def run_on_QC(self):
        pass

    def run(self, global_model):
        model = global_model
        model.train()
        # model.freeze_global_params()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print("client:",self.cid,"num_qubits:",self.num_qubits,"is training")
        for batch_idx, batch_data in enumerate(self.dataloader):
            if len(batch_data["image"]) <32:
                # 可能是这个原因
                break
            inputs = batch_data["image"].to(self.dev)
            target_all = batch_data["digit"].to(self.dev)
            outputs = model(inputs,qbit_idx=self.qbits_idx,use_qiskit=False,cid=self.cid,global_train=False)
            # target_all = torch.cat(targets, dim=0)
            # print(outputs)
            # print(target_all)
            loss = F.nll_loss(outputs, target_all)
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break
            # print(f"client:",self.cid,"loss:",{loss.item()}, end="\r")
        # self.scheduler.step()
        # return self.model.state_dict()