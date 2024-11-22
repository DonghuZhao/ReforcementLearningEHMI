from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import time
from config import *

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

        # 创建动态贝叶斯网络模型
        # D: distance
        # Dp: distance_to_stopline
        # Dv: delta_velocity
        # V: velocity
        # I -->  D
        #   -->  Dp   --> V
        #   -->  Dv
        # self.model = DBN([
        #     (('I', 0), ('D', 0)), (('I', 0), ('Dp', 0)), (('I', 0), ('Dv', 0)), (('Dp', 0), ('V', 0)),
        #     (('I', 0), ('I', 1)), (('D', 0), ('D', 1)), (('Dp', 0), ('Dp', 1))
        # ])
        # I -->  V    --> Dp
        #             --> D
        #   -->  Dv
        self.model = DBN([
            (('I', 0), ('V', 0)), (('V', 0), ('Dp', 0)), (('I', 0), ('Dv', 0)), (('V', 0), ('D', 0)),
            (('I', 0), ('I', 1)), (('D', 0), ('D', 1)), (('Dp', 0), ('Dp', 1))
        ])
        # 添加离散节点的 CPD
        top_node = ("I", 0)
        cpd = TabularCPD(variable=top_node, variable_card=len(discrete_node_values[top_node]),
                         values=[[0.5] for p in discrete_node_values[top_node]])
        self.model.add_cpds(cpd)
        # 为每个边添加 CPD
        # 根据模型结构，我们假设每个节点的取值都相互独立，即每个节点只受其父节点影响
        for node in self.model.nodes():
            parents = self.model.get_parents(node)
            if len(parents) == 1:
                parent = parents[0]
                # print("parent:", parent, "child:", node)
                parent_cardinality = len(discrete_node_values[parent])
                child_cardinality = len(discrete_node_values[node])
                cpd_values = np.random.rand(child_cardinality, parent_cardinality)  # 随机生成概率值
                cpd_values = cpd_values / cpd_values.sum(axis=0, keepdims=True)  # 将概率值归一化
                if node == ("I", 1):
                    cpd_values = [[1 - INTENTION_TRANS_PROB, INTENTION_TRANS_PROB],
                                  [INTENTION_TRANS_PROB, 1 - INTENTION_TRANS_PROB]]
                # print(cpd_values)
                cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values,
                                 evidence=[parent],
                                 evidence_card=[parent_cardinality])
                self.model.add_cpds(cpd)
            elif len(parents) == 2:
                # print("parent:", parents, "child:", node)
                parent_cardinality = [len(discrete_node_values[parent]) for parent in parents]
                child_cardinality = len(discrete_node_values[node])
                cpd_values = np.random.rand(child_cardinality,
                                            parent_cardinality[0] * parent_cardinality[1])  # 随机生成概率值
                cpd_values = cpd_values / cpd_values.sum(axis=0, keepdims=True)  # 将概率值归一化
                # print(cpd_values)
                cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values,
                                 evidence=[parent for parent in parents],
                                 evidence_card=[parent_cardinality[0], parent_cardinality[1]])
                self.model.add_cpds(cpd)
            elif len(parents) == 3:
                # print("parent:", parents, "child:", node)
                parent_cardinality = [len(discrete_node_values[parent]) for parent in parents]
                child_cardinality = len(discrete_node_values[node])
                cpd_values = np.random.rand(child_cardinality,
                                            parent_cardinality[0] * parent_cardinality[1] * parent_cardinality[
                                                2])  # 随机生成概率值
                cpd_values = cpd_values / cpd_values.sum(axis=0, keepdims=True)  # 将概率值归一化
                # print(cpd_values)
                cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values,
                                 evidence=[parent for parent in parents],
                                 evidence_card=[parent_cardinality[0], parent_cardinality[1],
                                                parent_cardinality[2]])
                self.model.add_cpds(cpd)

        # 检查模型的结构和参数是否正确
        print("------------check dbn model---------------")
        if self.model.check_model():
            print('bayesian model is correct')
        else:
            raise ValueError('bayesian model is not correct')

    def load_model(self):
        """load the dbn model from file"""
        if not self.file_path:
            raise ValueError("file path is not specified")
        with open(self.file_path, 'rb') as f:
            self.model = pickle.load(f)

    def train(self, train_data, save_model=False):
        """train the dbn model based on dataset"""

        self.createDBNModel()
        # 模型训练
        self.model.fit(train_data)
        # 将训练好的 DBN 模型保存到文件中
        if save_model:
            with open(self.file_path, 'wb') as f:
                pickle.dump(self.model, f)
        # 输出条件概率
        self.showCPD()
        # 输出图像
        self.diGraphShow()


    def evaluate(self, test_data):
        """evaluate the dbn model based on dataset"""

        # 推断部分
        if not self.model:
            with open(self.file_path, 'rb') as f:
                self.model = pickle.load(f)
        # 逐个轨迹推理
        for test_sample in test_data:
            start = time.time()
            self.infer_track(test_sample)
            cost = time.time() - start
            print(f"inferring track lenght {len(test_sample)}, cost: {cost}")

    def updateInference(self):
        """update DBN Inference based on new trained model"""
        try:
            if not self.model:
                raise ModuleNotFoundError("the dbn model to generate inference is None")
            # 创建DBN推断对象
            self.inference = DBNInference(self.model)
        except:
            raise ValueError("failed to generate inference")

    def inferring(self, track_part):
        """inferring the intention of the track"""
        if not self.inference:
            self.updateInference()
        # 修改列名
        merged_data_columns = []
        # print("evidence_len:", len(track_part))
        for time_ in range(len(track_part)):
            merged_data_columns.extend([('V', time_), ('D', time_), ('Dp', time_), ('Dv', time_)])
        merged_data = pd.DataFrame(track_part.values.reshape(1, -1), columns=merged_data_columns)
        # 创建观测值字典
        # print("-------------inferring-----------")
        evidence = merged_data.to_dict("index")[0]
        # print(evidence)

        # 推理目标
        last_index = HISTORY_LENGTH // STEP - 1
        res = []
        infer_target = [('I', time_ - ti) for ti in range(last_index, -1, -1)]
        result = self.inference.backward_inference(infer_target, evidence)
        for ti in range(last_index, -1, -1):
            temp = result[('I', time_ - ti)].values
            print("后向推断结果：", temp[1])
            res.append(temp[1])
        print(res)
        result = sum(res[1:-1]) / len(res[1:-1])
        return result

    def infer_track(self, track):
        """inferring the intention of the track"""
        # 更新推理模型
        self.updateInference()
        # 意图真值
        truth = track["I"].values[0]
        track = track[["V", "D", "Dp", "Dv"]]
        m, n = track.shape
        res = []
        uncertainty = []
        for t in range(m):
            if t < HISTORY_LENGTH // STEP:
                continue
            track_part = track[t - HISTORY_LENGTH // STEP + 1:t + 1]

            result = self.inferring(track_part)

            print(f"t: {t}, 最终意图推断结果：{result}")
            res.append(result)
            uncertainty.append(_calc_uncertainty(result))

        print("len_res:", len(res))
        if len(res):
            plt.plot(res, linewidth=5)
            # plt.plot(np.convolve(res, v=[0.3, 0.3, 0.4], mode="valid"), linewidth=5)
            # plt.plot(np.convolve(uncertainty, v = [0.3, 0.3, 0.4], mode = "valid"), linewidth = 2, color = 'orange')
            plt.title("Ground_truth:{}".format("YIELD" if truth else "PREEMPT"))
            plt.ylim([0, 1])
            plt.ylabel("Yield Probability", fontsize=18)
            plt.yticks(fontsize=15)
            plt.xlabel(f"timestamp/{0.1 * STEP}s", fontsize=18)
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

def _calc_uncertainty(p):
    """根据概率计算不确定性"""
    if p == 0 or p == 1:
        return 0
    if p < 0 or p > 1:
        raise ValueError("p should be in range [0, 1]")
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)