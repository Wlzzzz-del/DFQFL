import torch
import torch.nn.functional as F

import torchquantum as tq
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)
from components import Layer, Topology
import circuit_remapper as circuit_remapper


class QLayer(tq.QuantumModule):
    def __init__(self,n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        # 因为有这个random layer所以才会导致后面gates都不一样
        self.random_layer = tq.RandomLayer(
            n_ops=50, wires=list(range(self.n_wires))
        )

        # gates with trainable parameters
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)

        # some trainable gates (instantiated ahead of time)
        # 这些是trainable的gates
        # 我现在想的是能不能gates设计好之后保留，修改结构，重新计算,挪用到下一个结构,下一个结构CNOT与服务器?
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])

        # add some more non-parameterized gates (add on-the-fly)
        qdev.h(wires=3)  # type: ignore
        qdev.sx(wires=2)  # type: ignore
        qdev.cnot(wires=[3, 0])  # type: ignore
        qdev.rx(
            wires=1,
            params=torch.tensor([0.1]),
            static=self.static_mode,
            parent_graph=self.graph,
        )  # type: ignore

class QFCModel(tq.QuantumModule):

    def __init__(self,n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3_h_rx"])
        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )

        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        devi = x.device

        if use_qiskit:
            # use qiskit to process the circuit
            # create the qiskit circuit for encoder
            self.encoder(qdev, x)  
            op_history_parameterized = qdev.op_history
            qdev.reset_op_history()
            encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

            # create the qiskit circuit for trainable quantum layers
            self.q_layer(qdev)


            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

            # create the qiskit circuit for measurement
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)

            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )
            x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs).to(  # type: ignore
                devi
            )
            x = x0

            # 下面这些都是分布式的代码尝试，整理到remote model中
            # assemble the encoder, trainable quantum layers, and measurement circuits
            # the circs after transform can be run on qiskit processor

            # import copy
            # double_circs = qiskit_compose_circs([q_layer_circ,copy.deepcopy(q_layer_circ)])
            # assembled_circs1 = qiskit_assemble_circs(
            #     encoder_circs, double_circs, measurement_circ
            # )

            # print("len2",len(assembled_circs2))
            # assembled_circs2[0].draw(output='mpl')
            # plt.show()
            # plt.clf()

            # call the qiskit processor to process the circuit
            # 缺少了定义qiskit_processor
            # 但我这边直接尝试distribution
            # circuit_topo = Topology()
            # # 创建两个QPU，每个QPU有2个qubit
            # circuit_topo.create_qmap(3, [4, 4, 4],"q")

            # import matplotlib.pyplot as plt
            # # qregs = circuit_topo.get_regs()
            # remapper = circuit_remapper.CircuitRemapper(circuit_topo)
            # dist_circ = remapper.remap_circuit(assembled_circs[0],True)

            # 输出的图没区别..
            # assembled_circs[0].draw(output='mpl')
            # dist_circ.draw(output='mpl')
            # plt.show()

            # print("successfully build")

        else:
            # use torchquantum to process the circuit
            self.encoder(qdev, x)
            qdev.reset_op_history()
            self.q_layer(qdev)
            x = self.measure(qdev)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x