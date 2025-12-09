import torch.optim as optim
import torch.nn.functional as F
import torch

class SUB_QFL_Client():
    def __init__(self,cid,qbits_idx,dataloader,lr,model,dev,epochs,num_qubits,batch_size=32,weight_decay=1e-4):
        self.batch_size=batch_size
        self.cid = cid
        self.qbits_idx = qbits_idx
        self.dataloader = dataloader
        self.lr = lr
        self.model = model
        self.dev = dev
        self.local_epochs = epochs
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.num_qubits = num_qubits

    def run_on_QC(self):
        # run on quantum computer
        pass

    def run(self, global_model):
        # run on classical computer
        model = global_model
        model.train()
        for _epoch in range(self.local_epochs):
            for batch_idx, batch_data in enumerate(self.dataloader):
                inputs = batch_data[0].to(self.dev)
                target_all = batch_data[1].to(self.dev)
                outputs = model(inputs,qbit_idx=self.qbits_idx,use_qiskit=False,cid=self.cid,global_train=False)
                loss = F.nll_loss(outputs, target_all)
                # print(loss.item())
                self.optimizer.zero_grad()
                loss.backward()

                # # TOBEFIX:好像对1来说，0的电路也参与了本地计算
                # print("--- 检查梯度 ---")
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         if param.grad is None:
                #             print(f"❌ {name}: 梯度为 None (计算图中断或未参与计算)")
                #         elif torch.sum(param.grad) == 0:
                #             print(f"⚠️ {name}: 梯度全为 0 (可能是 Dead ReLU 或 落入平坦区域)")
                #         else:
                #             # 打印梯度的平均值或最大值来确认
                #             print(f"✅ {name}: 梯度正常 (Mean: {param.grad.abs().mean().item()})")

                self.optimizer.step()