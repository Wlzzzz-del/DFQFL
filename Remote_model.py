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
# from sub_encoder import MyEncoder
import copy
import math
from torchquantum import GeneralEncoder


class Automated_QLayer(tq.QuantumModule):
    def __init__(self, n_wires, wires):
        super().__init__()
        self.n_wires = n_wires
        self.wires = wires
        # 100有点过拟合了
        self.random_nops = 80# 50的效果还不错
        # 添加非参数门试一下.......

        # --- 1. 自动生成单比特旋转门 (Single Qubit Gates) ---
        # 使用 QuantumModuleList 来存储列表中的层，这样参数才能被优化器识别
        self.rx_gates = tq.QuantumModuleList()
        self.ry_gates = tq.QuantumModuleList()
        self.rz_gates = tq.QuantumModuleList()

        for _ in range(n_wires):
            self.rx_gates.append(tq.RX(has_params=True, trainable=True))
            self.ry_gates.append(tq.RY(has_params=True, trainable=True))
            self.rz_gates.append(tq.RZ(has_params=True, trainable=True))

        # --- 2. 自动生成纠缠门 (Entangling Gates) ---
        # 这里实现一个“最近邻纠缠” (Nearest Neighbor Entanglement)
        self.crx_gates = tq.QuantumModuleList()

        # 如果只有1个qubit，就不需要纠缠门
        if n_wires > 1:
            for _ in range(n_wires - 1):
                self.crx_gates.append(tq.CRX(has_params=True, trainable=True))

            # 如果你想形成环状纠缠 (Ring)，可以把下面这行解开，连接最后一个和第一个
            # self.crx_gates.append(tq.CRX(has_params=True, trainable=True))

        # 随机层逻辑 (可选)
        self.random_layer = tq.RandomLayer(n_ops=self.random_nops, wires=wires)

    def forward(self, qdev: tq.QuantumDevice, wires=None):
        # 如果调用时没指定wires，就用初始化时的wires
        if wires is None:
            wires = self.wires

        # 1. 应用随机层 (如果需要)
        self.random_layer(qdev)

        # 2. 动态应用单比特门
        # 逻辑：遍历每个qubit，依次应用 RX, RY, RZ
        for i in range(self.n_wires):
            # 确保不越界，虽然通常 wires 长度应等于 n_wires
            if i < len(wires):
                wire = wires[i]
                qdev.h(wire)
                qdev.sx(wire)
                qdev.rx(
                    wires=wire,
                    params=torch.tensor([0.1]),
                    static=self.static_mode,
                    parent_graph=self.graph,)
                self.rx_gates[i](qdev, wires=wire)
                self.ry_gates[i](qdev, wires=wire)
                self.rz_gates[i](qdev, wires=wire)

        # 3. 动态应用纠缠门 (CRX)
        # 逻辑：控制比特是 wires[i]，目标比特是 wires[i+1]
        if self.n_wires > 1:
            for i in range(len(self.crx_gates)):
                # 获取当前CRX门
                crx_op = self.crx_gates[i]
                # 定义连接关系：当前qubit控制下一个qubit
                control_wire = wires[i]
                target_wire = wires[(i + 1) % len(wires)] # 使用取模防止索引越界

                # 应用CRX门
                crx_op(qdev, wires=[control_wire, target_wire])

        # 4. 动态添加非参数化门 (On-the-fly)
        # 这里演示如何根据循环添加 CNOT
        for i in range(self.n_wires - 1):
             qdev.cnot(wires=[wires[i+1], wires[i]]) # 反向CNOT链

class Remoted_QFCModel(tq.QuantumModule):

    def __init__(self,wires,n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = GeneralEncoder(tq.encoder_op_list_name_dict[self.find_encoder_dict_key()])
        self.q_layer = Automated_QLayer(len(wires),wires)

    def find_encoder_dict_key(self):
        """Return the key for encoder op list dict according to n_wires.

        Returns:
            _type_: _description_
        """
        if self.n_wires == 1:
            return "1x1_ry"
        elif self.n_wires == 2:
            return "2x8_rxryrzrxryrzrxry"
        elif self.n_wires == 3:
            return "3x1_rxrxrx"
        elif self.n_wires == 4:
            return "4x4_ryzxy"
        elif self.n_wires == 8:
            return "8x2_ry"

    def copy_encoder_params(self, x, target_dev,adding):
        """
        build new dev and copy encoder params to target_dev
        copy encoder params from small qubit dev to large qubit dev by padding zeros
        Args:
            x (_type_): _input data
            target_dev (_type_): _target quantum device
            adding (_type_): _client qubit adding number

        Returns:
            _type_: _copied states
        """
        batch_size = x.shape[0]
        new_qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=batch_size, device=x.device, record_op=True
        )
        self.encoder(new_qdev, x)# encoded to small dev
        small_state = new_qdev.states
        large_state = target_dev.states

        tensor_small = small_state
        tensor_large = large_state

        # 2. Flatten
        flat_small = tensor_small.reshape(batch_size, -1)
        flat_large = tensor_large.reshape(batch_size, -1)

        # 3. Number of padding zeros
        diff = flat_large.shape[1] - flat_small.shape[1] # 1024 - 8 = 1016

        # 4. Pad zeros
        padded_small = F.pad(flat_small, (int(math.pow(2,adding)), diff-int(math.pow(2,adding))), "constant", 0)

        # 5. Adding
        result_flat = flat_large + padded_small

        result = result_flat.view(tensor_large.shape)
        # 6. clean up
        del new_qdev
        del padded_small
        return result

    def forward(self,x, qdev, wires, local_train=False,use_qiskit=False):

        bsz = x.shape[0]

        x = x.view(-1, 1, 28, 28)
        x = F.avg_pool2d(x, 6).view(bsz, -1)

        if not use_qiskit:
            if local_train:
                # use torchquantum to process the circuit
                # 有两个思路，一个是重新修改自己的encoder,以支持wires的输入，另一个思路是看qdev能不能拆分出来
                # print("DEBUG: Local TRAINING")
                # _adding = wires[0]# 客户端qubit的增量
                # qdev.set_states(self.copy_encoder_params(x, qdev, _adding))
                # print("DEBUG: Local TRAINING with wires adjustment")
                # _start = wires[0]
                # wires = [wires[i]-_start for i in range(len(wires))]
                # print(wires)
                self.encoder(qdev, x)
                qdev.reset_op_history()
                self.q_layer(qdev)
            else:
                _adding = wires[0]# 客户端qubit的增量
                qdev.set_states(self.copy_encoder_params(x, qdev, _adding))
                self.q_layer(qdev, wires)


        else:
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

            # assemble the encoder, trainable quantum layers, and measurement circuits
            # the circs after transform can be run on qiskit processor
            # import copy
            # double_circs = qiskit_compose_circs([q_layer_circ,copy.deepcopy(q_layer_circ)])
            # assembled_circs1 = qiskit_assemble_circs(
            #     encoder_circs, double_circs, measurement_circ
            # )

            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )
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
            # x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs).to(  # type: ignore
            #     devi
            # )
            # x = x0

