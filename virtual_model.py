from utils import qiskit_compose_circs
import torch
import torch.nn.functional as F
from qiskit import QuantumRegister
import torchquantum as tq
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)
from utils import expanded_imgs_for_wires

class VirtualGlobalModel(tq.QuantumModule):
    def __init__(self, qmodel_lists, n_qubits, qubits_list):
        super().__init__()
        self.n_wires = n_qubits
        self.c_model_list = tq.QuantumModuleList()
        for i, qmodel in enumerate(qmodel_lists):
            self.c_model_list.append(qmodel)
        self.crxs = [tq.CRX(has_params=True, trainable=True) for _i in range(len(qubits_list))]
        self.qubits_list = qubits_list
        self.measure = tq.MeasureAll(tq.PauliZ)

    def build_global_entangle(self, qdev: tq.QuantumDevice):
        # qdev: Quantum device with all client qubits
        # build global entangle layer params between different clients
        wirs = 0
        count = 0
        for wir in self.qubits_list:
            qdev.h(wires=wir)  # type: ignore
            qdev.sx(wires=wir)
            if wirs+wir == self.n_wires:
                self.crxs[count](qdev, wires=[0, self.n_wires - 1])
                break
            self.crxs[count](qdev, wires=[wirs+wir-1,wirs+wir])
            wirs += wir
            count += 1
        return qdev

    def distribute_client_qubits(self):
        # distribute qubits to clients
        # Output: cid -> qubits list
        usr_to_qubits = dict()
        point = 0
        for cid in range(len(self.qubits_list)):
            usr_to_qubits[cid] = list(range(point, point+self.qubits_list[cid]))
            point += self.qubits_list[cid]
        return usr_to_qubits

    def freeze_global_params(self):
        # freeze global entangle params and make it not trainable
        for crx in self.crxs:
            for name,para in crx.named_parameters():
                para.requires_grad = False

    def frozen_global_params(self):
        # freeze global entangle params and make it trainable
        for crx in self.crxs:
            for name,para in crx.named_parameters():
                para.requires_grad = True

    def forward(self, x, qbit_idx=[],use_qiskit=False,cid=-1,global_train=True):

        if use_qiskit:
            pass
        else:
            # if train virtual global model with global entangle layer
            if global_train == True:
                bsz = len(x[0][0])*len(x)
                # print(bsz)
                qdev = tq.QuantumDevice(
                    n_wires=self.n_wires, bsz=bsz, device=x[0][0].device, record_op=True
                )
                qdev = self.build_global_entangle(qdev)
                self.usr_to_qubits = self.distribute_client_qubits()

                targets = []

                # TOBEFIX: 这边的逻辑能不能改进？
                for cid in range(len(x)):
                    batch = x[cid]
                    qmodel = self.c_model_list[cid]

                    # expanded_imgs = expanded_imgs_for_wires(batch[0],len(self.usr_to_qubits))
                    expanded_imgs = batch[0]  # cid要去掉
                    qmodel(expanded_imgs, qdev, wires=self.usr_to_qubits[cid])# wires需要指示该客户端有哪些qubits
                    targets.append(batch[1])
                x = self.measure(qdev)
                x = F.log_softmax(x, dim=1)

            # Perform local training on clients circuits only
            else:
                bsz = x.shape[0]
                # bsz = x.shape[0]*len(self.c_model_list)
                self.usr_to_qubits = self.distribute_client_qubits()
                qdev = tq.QuantumDevice(
                    n_wires=len(qbit_idx), bsz=bsz, device=x.device, record_op=True
                )
                batch = x# cid要去掉
                model = self.c_model_list[cid]# cid 要去掉
                # model.zero_grad()
                expanded_imgs = batch
                model(expanded_imgs, qdev, local_train=True, wires=self.usr_to_qubits[cid])# wires需要指示该客户端有哪些qubits
                x = self.measure(qdev)
                x = F.log_softmax(x, dim=1)
                return x

            return x,targets