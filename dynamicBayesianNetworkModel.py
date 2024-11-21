from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

# HyperParams
INTENTION_TRANS_PROB = 0.1

class dynamicBayesianNetworkModel:
    """
    Dynamic Bayesian Network Model
    file_path: path to save the dynamic bayesian network model;
    model: dynamic bayesian network model;
    inference: inference generate from dbn model;
    """
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.model = None
        self.inference = None

    def createDBNModel(self):
        """create DBN model by define nodes, edges and initiate CPD value"""
        print("---------------------create dbn model----------------------")
        # 定义节点的取值范围和间隔
        # 节点的取值范围离散化
        I_values = [0, 1]
        Dp_values = list(range(-20, 61, 5))
        Dv_values = list(range(-10, 11))
        V_values = list(range(0, 21, 2))
        D_values = list(range(0, 61, 5))
        # 定义每个节点的取值范围
        discrete_node_values = {
            ('I', 0): I_values,
            ('D', 0): D_values,
            ('Dp', 0): Dp_values,
            ('Dv', 0): Dv_values,
            ('V', 0): V_values,
            ('I', 1): I_values,
            ('D', 1): D_values,
            ('Dp', 1): Dp_values,
            ('Dv', 1): Dv_values,
            ('V', 1): V_values
        }

        # TODO 创建动态贝叶斯网络模型
        # D: distance
        # Dp: distance_to_stopline
        # Dv: delta_velocity
        # V: velocity
        # I -->  D
        #   -->  Dp   --> V
        #   -->  Dv
        dbn = DBN([
            (('I', 0), ('D', 0)), (('I', 0), ('Dp', 0)), (('I', 0), ('Dv', 0)), (('Dp', 0), ('V', 0)),
            (('I', 0), ('I', 1)), (('D', 0), ('D', 1)), (('Dp', 0), ('Dp', 1))
        ])
        # 添加离散节点的 CPD
        top_node = ("I", 0)
        cpd = TabularCPD(variable=top_node, variable_card=len(discrete_node_values[top_node]),
                         values=[[0.5] for p in discrete_node_values[top_node]])
        dbn.add_cpds(cpd)
        # 为每个边添加 CPD
        # 根据模型结构，我们假设每个节点的取值都相互独立，即每个节点只受其父节点影响
        for node in dbn.nodes():
            parents = dbn.get_parents(node)
            if len(parents) == 1:
                parent = parents[0]
                print("parent:", parent, "child:", node)
                parent_cardinality = len(discrete_node_values[parent])
                child_cardinality = len(discrete_node_values[node])
                cpd_values = np.random.rand(child_cardinality, parent_cardinality)  # 随机生成概率值
                cpd_values = cpd_values / cpd_values.sum(axis=0, keepdims=True)  # 将概率值归一化
                if node == ("I", 1):
                    cpd_values = [[1 - INTENTION_TRANS_PROB, INTENTION_TRANS_PROB],
                                  [INTENTION_TRANS_PROB, 1 - INTENTION_TRANS_PROB]]
                print(cpd_values)
                cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values,
                                 evidence=[parent],
                                 evidence_card=[parent_cardinality])
                dbn.add_cpds(cpd)
            elif len(parents) == 2:
                print("parent:", parents, "child:", node)
                parent_cardinality = [len(discrete_node_values[parent]) for parent in parents]
                child_cardinality = len(discrete_node_values[node])
                cpd_values = np.random.rand(child_cardinality,
                                            parent_cardinality[0] * parent_cardinality[1])  # 随机生成概率值
                cpd_values = cpd_values / cpd_values.sum(axis=0, keepdims=True)  # 将概率值归一化
                print(cpd_values)
                cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values,
                                 evidence=[parent for parent in parents],
                                 evidence_card=[parent_cardinality[0], parent_cardinality[1]])
                dbn.add_cpds(cpd)
            elif len(parents) == 3:
                print("parent:", parents, "child:", node)
                parent_cardinality = [len(discrete_node_values[parent]) for parent in parents]
                child_cardinality = len(discrete_node_values[node])
                cpd_values = np.random.rand(child_cardinality,
                                            parent_cardinality[0] * parent_cardinality[1] * parent_cardinality[
                                                2])  # 随机生成概率值
                cpd_values = cpd_values / cpd_values.sum(axis=0, keepdims=True)  # 将概率值归一化
                print(cpd_values)
                cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values,
                                 evidence=[parent for parent in parents],
                                 evidence_card=[parent_cardinality[0], parent_cardinality[1],
                                                parent_cardinality[2]])
                dbn.add_cpds(cpd)

        # 检查模型的结构和参数是否正确
        print("------------check dbn model---------------\n", dbn.check_model())

        return dbn

    def load_model(self):
        """load the dbn model from file"""
        if not self.file_path:
            raise ValueError("file path is not specified")
        with open(self.file_path, 'rb') as f:
            self.model = pickle.load(f)

    def train(self, train_data, save_model=False):
        """train the dbn model based on dataset"""

        dbn = self.model
        # 模型训练
        dbn.fit(train_data)
        # 将训练好的 DBN 模型保存到文件中
        if save_model:
            with open(self.file_path, 'wb') as f:
                pickle.dump(dbn, f)
        # 输出条件概率
        self.showCPD()
        # 输出图像
        self.diGraphShow()

        return dbn

    def evaluate(self, test_data):
        """evaluate the dbn model based on dataset"""

        # 推断部分
        if not self.model:
            with open(self.file_path, 'rb') as f:
                self.model = pickle.load(f)

        # 输出图像
        self.diGraphShow()

        for test_sample in test_data:
            self.inferring(test_sample)

    def updateInference(self):
        """update DBN Inference based on new trained model"""
        try:
            if not self.model:
                raise ModuleNotFoundError("the dbn model to generate inference is None")
            # 创建DBN推断对象
            self.inference = DBNInference(self.model)
        except:
            raise ValueError("failed to generate inference")

    def inferring(self, track):
        """inferring the intention of the track"""
        self.updateInference()
        truth = track["I"].values[0]
        track = track[["V", "D", "Dp", "Dv"]]
        # track = track[["s_vx", "distance", "s_dp", "Dv"]]
        print(track.columns)
        m, n = track.shape
        res = []
        uncertainty = []
        for t in range(0, m, 5):
            if t == 0 or t == 5 or t == 10 or t == 15:
                continue
            print("t:", t)
            track_part = track[t - 19:t + 1]
            # 修改列名
            merged_data_columns = []
            print("evidence_len:", len(track_part))
            for time in range(len(track_part)):
                merged_data_columns.extend([('V', time), ('D', time), ('Dp', time), ('Dv', time)])
            merged_data = pd.DataFrame(track_part.values.reshape(1, -1), columns=merged_data_columns)
            # print(merged_data)
            # 创建观测值字典
            # print("-------------inferring-----------")
            evidence = merged_data.to_dict("index")[0]
            # print(evidence)

            # 进行后向推断并获取结果
            result = self.inference.backward_inference([('I', time - ti) for ti in range(4, -1, -1)], evidence)
            for ti in range(4, -1, -1):
                temp = result[('I', time - ti)].values
                print("后向推断结果：", temp[0])
                res.append(temp[0])
                uncertainty.append(-temp[0] * np.log2(temp[0]) - temp[1] * np.log2(temp[1]))
        print("len_res:", len(res))
        plt.plot(np.convolve(res, v=[0.3, 0.3, 0.4], mode="valid"), linewidth=5)
        # plt.plot(np.convolve(uncertainty, v = [0.3, 0.3, 0.4], mode = "valid"), linewidth = 2, color = 'orange')
        plt.title("Ground_truth:{}".format("PREEMPT" if truth else "YIELD"))
        plt.ylim([0, 1])
        plt.ylabel("Yield Probability", fontsize=18)
        plt.yticks(fontsize=15)
        plt.xlabel("timestamp/0.1s", fontsize=18)
        plt.xticks(fontsize=15)
        plt.legend(["YieldProb", "Uncertainty"], fontsize=18)
        plt.show()

    def showCPD(self):
        """show the CPD(conditional probability distribution) result of dbn model"""
        print('---------------------show CPD result of dbn model----------------------')
        dbn = self.model
        # 获取CPDs并转换为DataFrame
        cpd_data = []
        for cpd in dbn.get_cpds():
            cpd_data.append({
                "variable": cpd.variable,
                "cardinality": cpd.variable_card,
                "values": cpd.values
            })
        cpd_df = pd.DataFrame(cpd_data)
        print(cpd_df)
        return

    def diGraphShow(self):
        """show the topology of dbn model"""
        # 创建图对象
        G = nx.DiGraph()
        # 添加节点和边
        G.add_edges_from(self.model.edges())
        # 绘制拓扑关系图
        plt.figure(figsize=(10, 6))
        # 选择不同的布局方法
        # pos = nx.spring_layout(G)  # 弹簧布局
        pos = nx.circular_layout(G)  # 环形布局
        # pos = nx.spectral_layout(G)  # 谱布局
        # 调整节点大小和间距
        options = {
            "node_size": 2000,
            "node_color": "skyblue",
            "font_size": 12,
            "font_weight": "bold",
            "arrowsize": 20,
            "width": 2,
            "edge_color": "gray"
        }
        nx.draw(G, pos, with_labels=True, **options)
        plt.title("Dynamic Bayesian Network Topology", fontsize=15)
        plt.show()
