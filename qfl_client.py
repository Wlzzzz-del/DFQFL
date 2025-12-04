import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

class QFL_Client():
    def __init__(self,cid,dataloader,lr,model,dev,epochs):
        self.cid = cid
        self.dataloader = dataloader
        self.lr = lr
        self.model = model
        self.dev = dev
        self.local_epochs = epochs
        self.weight_decay = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.local_epochs)

    def run_on_QC(self):
        pass

    def run(self):
        self.model.train()
        # self.model.q_layer.static_on(wires_per_block=2)
        self.model.to(self.dev)
        for batch_idx, batch_data in enumerate(self.dataloader):
            inputs = batch_data["image"].to(self.dev)
            targets = batch_data["digit"].to(self.dev)
            outputs = self.model(inputs)
            loss = F.nll_loss(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"client:",self.cid,"loss:",{loss.item()}, end="\r")
        self.scheduler.step()
        return self.model.state_dict()