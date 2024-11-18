from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import os

# HyperParams
INTENTION_TRANS_PROB = 0.1


def diGraphShow(dbn):
    # 创建图对象
    G = nx.DiGraph()
    # 添加节点和边
    G.add_edges_from(dbn.edges())
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
    return

# 目标格式
class Track_info:
    def __init__(self):
        self.object_type = None
        self.track_id = None
        self.timestep = []
        self.position = []
        self.velocity = []
        self.heading = []
        self.observed = []
        self.category = None
        self.track_type = None
        self.s = []
        self.l = []

def trainDataTake():
    straight_col = [10, 16, 18, 19]
    left_col = [3, 16, 17, 18]
    # 训练数据
    meta_path = r"C:\Users\ZDH\OneDrive - tongji.edu.cn\硕士\代码\数据\SIND模糊场景数据\fuzzy_state_tracks_70组数据\fuzzy_state_meta.xlsx"
    # 提取需要的交互对
    meta = pd.read_excel(meta_path,sheet_name="Sheet1",header=0,skiprows=[1])
    meta = meta[meta["ego_entrance"] == 1 ] # 左转车进口道为西进口道
    meta = meta[meta["sig_state"] != 0 ]    # 信号灯为绿灯
    # 轨迹存储路径
    new_path = r"C:\Users\ZDH\OneDrive - tongji.edu.cn\硕士\代码\数据\SIND模糊场景数据\fuzzy_state_tracks"

    train_data_combined = pd.DataFrame()
    test_data_combined = pd.DataFrame()
    test_set = []
    # 将分散的交互对合并成一个训练数据集
    for index,sample in meta.iterrows():
        # if sample["pass_seq"] == 1:
        #     continue
        _id = int(sample["id"])
        print(_id, "read", "---result:", sample["pass_seq"])
        folder = os.path.join(new_path,str(_id))

        data_file = os.path.join(folder,"both.csv")
        with open(data_file, 'rb') as filehandle:
            train_data = pd.read_csv(data_file, header = 0, usecols= left_col)

        # 数据编辑
        train_data.rename(columns={"Dv": "Dv", "l_dp": "Dp", "distance": "D",
                                   "l_vx": "V"}, inplace=True)
        # train_data.rename(columns = {"Dv":"Dv", "s_dp":"Dp","distance":"D",
        #                              "s_vx":"V"}, inplace=True)
        train_data = dataDiscretization(train_data)
        if len(train_data.index) % 2 ==1 :
            train_data = train_data[:-1]
        train_data["I"] = sample["pass_seq"]    # if pass_seq == 1 左转先行， 对应直行车的 Yield_probability
        train_data = train_data.astype(int)
        # print(train_data)
        if _id in [35, 62, 17]:   # 35 & 62, 17
            test_data_combined = pd.concat([test_data_combined, train_data])
            test_set.append(test_data_combined)
        else:
            train_data = dataStructTransform(train_data)
            train_data_combined = pd.concat([train_data_combined, train_data])

    # 增加训练代数
    for i in range(10):
        train_data_combined = pd.concat([train_data_combined, train_data_combined])
    print(train_data_combined)
    return train_data_combined, test_set

def trainDataTake2():
    straight_col = [10, 16, 18, 19]
    left_col = [3, 16, 17, 18]
    # 训练数据
    meta_path = r"C:\Users\ZDH\OneDrive - tongji.edu.cn\硕士\代码\数据\驾驶模拟实验数据\格式化数据\total.csv"
    # 提取需要的交互对
    meta = pd.read_csv(meta_path,header=0)

    # 轨迹存储路径
    new_path = r"C:\Users\ZDH\OneDrive - tongji.edu.cn\硕士\代码\数据\驾驶模拟实验数据\格式化数据"

    train_data_combined = pd.DataFrame()
    test_data_combined = pd.DataFrame()
    test_set = []
    # 将分散的交互对合并成一个训练数据集
    for index,sample in meta.iterrows():
        driver = int(sample["driver_id"])
        scenerio = int(sample["scenerio_id"])
        print("driver{}scenerio{}".format(driver, scenerio), "read", "---result:", sample["ego_seq"])
        folder = os.path.join(new_path,"driver_{}".format(driver),"scenerio_{}".format(scenerio))

        data_file = os.path.join(folder,"both.csv")
        with open(data_file, 'rb') as filehandle:
            train_data = pd.read_csv(data_file, header = 0, usecols= left_col)

        # 数据编辑
        train_data.rename(columns={"Dv": "Dv", "l_dp": "Dp", "distance": "D",
                                   "l_vx": "V"}, inplace=True)
        # train_data.rename(columns = {"Dv":"Dv", "s_dp":"Dp","distance":"D",
        #                              "s_vx":"V"}, inplace=True)
        train_data = dataDiscretization(train_data)
        if len(train_data.index) % 2 ==1 :
            train_data = train_data[:-1]
        train_data["I"] = sample["ego_seq"]    # if pass_seq == 1 左转先行， 对应直行车的 Yield_probability
        train_data = train_data.astype(int)
        train_data = dataStructTransform(train_data)
        # print(train_data)
        train_data_combined = pd.concat([train_data_combined, train_data])

    # 增加训练代数
    for i in range(10):
        train_data_combined = pd.concat([train_data_combined, train_data_combined])
    print(train_data_combined)
    return train_data_combined, test_set

def dataDiscretization(train_data):
    Dp_values = list(range(-20, 61, 5))
    Dv_values = list(range(-10, 11))
    V_values = list(range(0, 21, 2))
    D_values = list(range(0, 61, 5))
    train_data["Dp"] = train_data["Dp"].apply(continousToDiscrete, criteria = Dp_values)
    train_data["Dv"] = train_data["Dv"].apply(continousToDiscrete, criteria = Dv_values)
    train_data["V"] = train_data["V"].apply(continousToDiscrete, criteria = V_values)
    train_data["D"] = train_data["D"].apply(continousToDiscrete, criteria = D_values)
    return train_data
def dataStructTransform(data):
    # 将 DataFrame 转换为一行并添加索引到列名
    colnames = data.columns
    m, n = data.shape
    merged_data = data.stack().to_frame().T
    # 行复制
    merged_data = pd.DataFrame(np.repeat(merged_data.values, repeats = 2, axis = 0))[1:-1]
    # 修改列名
    merged_data_columns = []
    for t in range(2):
        merged_data_columns.extend([('V', t), ('D', t), ('Dp', t), ('Dv', t), ('I', t)])
    merged_data = pd.DataFrame(data.values.reshape(-1, 10), columns=merged_data_columns)
    # for idx, col in enumerate(merged_data.columns):
    #     merged_data_columns.append((col[1],col[0]))
    # # 显示合并后的数据
    # merged_data = pd.DataFrame(data.values.reshape(1, -1), columns=pd.MultiIndex.from_tuples(merged_data_columns))
    # print(merged_data.columns)
    # print(merged_data)
    return merged_data

def showCPD(dbn):
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

def createDBNModel():
    # 定义节点的取值范围和间隔
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
    dbn = DBN([
        (('I', 0), ('D', 0)), (('I', 0), ('Dp', 0)), (('I', 0), ('Dv', 0)), (('Dp', 0), ('V', 0)),
        (('I', 0), ('I', 1)), (('D', 0), ('D', 1)), (('Dp', 0), ('Dp', 1))
    ])
    # 添加离散节点的 CPD
    top_node = ("I", 0)
    cpd = TabularCPD(variable=top_node, variable_card=len(discrete_node_values[top_node]), values=[[0.5] for p in discrete_node_values[top_node]])
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
                cpd_values = [[1 - INTENTION_TRANS_PROB, INTENTION_TRANS_PROB], [INTENTION_TRANS_PROB, 1 - INTENTION_TRANS_PROB]]
            print(cpd_values)
            cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values, evidence=[parent],
                             evidence_card=[parent_cardinality])
            dbn.add_cpds(cpd)
        elif len(parents) == 2:
            print("parent:", parents, "child:", node)
            parent_cardinality = [len(discrete_node_values[parent]) for parent in parents]
            child_cardinality = len(discrete_node_values[node])
            cpd_values = np.random.rand(child_cardinality, parent_cardinality[0] * parent_cardinality[1])  # 随机生成概率值
            cpd_values = cpd_values / cpd_values.sum(axis=0, keepdims=True)  # 将概率值归一化
            print(cpd_values)
            cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values, evidence=[parent for parent in parents],
                             evidence_card=[parent_cardinality[0], parent_cardinality[1]])
            dbn.add_cpds(cpd)
        elif len(parents) == 3:
            print("parent:", parents, "child:", node)
            parent_cardinality = [len(discrete_node_values[parent]) for parent in parents]
            child_cardinality = len(discrete_node_values[node])
            cpd_values = np.random.rand(child_cardinality, parent_cardinality[0] * parent_cardinality[1] * parent_cardinality[2])  # 随机生成概率值
            cpd_values = cpd_values / cpd_values.sum(axis=0, keepdims=True)  # 将概率值归一化
            print(cpd_values)
            cpd = TabularCPD(variable=node, variable_card=child_cardinality, values=cpd_values, evidence=[parent for parent in parents],
                             evidence_card=[parent_cardinality[0], parent_cardinality[1], parent_cardinality[2]])
            dbn.add_cpds(cpd)


    # 检查模型的结构和参数是否正确
    print(dbn.check_model())
    # print(dbn.check_model())
    return dbn

def inferring(dbn, track):
    truth = track["I"].values[0]
    track = track[["V","D","Dp","Dv"]]
    # track = track[["s_vx", "distance", "s_dp", "Dv"]]
    print(track.columns)
    m, n = track.shape
    res = []
    uncertainty =[]
    for t in range(0, m, 5):
        if t == 0 or t == 5 or t == 10 or t == 15:
            continue
        print("t:", t)
        track_part = track[t - 20:t + 1]
        # 修改列名
        merged_data_columns = []
        print("evidence_len:", len(track_part))
        for time in range(len(track_part)):
            merged_data_columns.extend([('V', time), ('D', time), ('Dp', time), ('Dv', time) ])
        merged_data = pd.DataFrame(track_part.values.reshape(1, -1), columns=merged_data_columns)
        # print(merged_data)
        # 创建观测值字典
        # print("-------------inferring-----------")
        evidence = merged_data.to_dict("index")[0]
        # print(evidence)
        # 创建后向推断对象
        dbn_inf = DBNInference(dbn)

        # 进行后向推断并获取结果
        result = dbn_inf.backward_inference([('I', time - ti) for ti in range(4, -1, -1)], evidence)
        for ti in range(4, -1, -1):
            temp = result[('I', time - ti)].values
            print("后向推断结果：", temp[0])
            res.append(temp[0])
            uncertainty.append(-temp[0] * np.log2(temp[0]) - temp[1] * np.log2(temp[1]))
    print("len_res:",len(res))
    plt.plot(np.convolve(res, v = [0.3, 0.3, 0.4], mode = "valid"), linewidth = 5)
    # plt.plot(np.convolve(uncertainty, v = [0.3, 0.3, 0.4], mode = "valid"), linewidth = 2, color = 'orange')
    plt.title("Ground_truth:{}".format("PREEMPT" if truth else "YIELD"))
    plt.ylim([0,1])
    plt.ylabel("Yield Probability", fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xlabel("timestamp/0.1s", fontsize = 18)
    plt.xticks(fontsize = 15)
    plt.legend(["YieldProb", "Uncertainty"],fontsize = 18)
    plt.show()
    return

def continousToDiscrete(variable, criteria):
    for i in range(len(criteria)):
        if variable < criteria[i]:
            break
    return i

def testDataTake():
    testData_set = []
    for driver in range(4,5):
        for scenerio in range(10,15):
            file_path = f"C:\\Users\\ZDH\\OneDrive\\交互行为\\SILAB第三次实验\\驾驶数据\\驾驶员{driver + 1}\\DataFile." + str(
                scenerio + 1) + ".asc"
            data_all = pd.read_csv(file_path)
            print("read", scenerio)

            col_names = []
            col_names.append("Straight_X")
            col_names.append("Straight_Y")
            col_names.append("Straight_yaw")
            col_names.append("Straight_v_km/h")
            col_names.append("Left_X")
            col_names.append("Left_Y")
            col_names.append("Left_yaw")
            col_names.append("V1")
            col_names.append("K1.NumPad1")
            col_names.append("K1.NumPad2")
            col_names.append("Straight_longterm_decision_flag")
            data_single = data_all[col_names]
            NumPad1 = np.where(data_single["K1.NumPad1"] != 0)[0]
            NumPad2 = np.where(data_single["K1.NumPad2"] != 0)[0]
            if NumPad1.any():
                intention_recognition = NumPad1[0]
                intention = "PREEMPT"
            else:
                intention_recognition = NumPad2[0]
                intention = "YILED"
            print("intention:", intention)
            print("intention_recognition", intention_recognition)
            data_single["Straight_X"] = 266.7007 - data_single["Straight_X"]
            data_single = data_single[intention_recognition - 20:intention_recognition + 25]
            data_single["s_x"] = data_single["Straight_X"]
            data_single["s_vx"] = abs(data_single["Straight_v_km/h"] / 3.6)
            data_single["distance"] = abs(data_single["Straight_X"] - data_single["Left_X"])
            data_single["l_vx"] = abs(data_single["V1"])
            testData_set.append(dataDiscretization(data_single[["s_x","s_vx","distance","l_vx"]]))
    return testData_set

def train(train_data, save_model=False):
    """train the dbn model based on dataset"""

    # 模型构建
    dbn = createDBNModel()
    # 模型训练
    dbn.fit(train_data)
    # 将训练好的 DBN 模型保存到文件中
    if save_model:
        with open(file_path, 'wb') as f:
            pickle.dump(dbn, f)
    # 输出条件概率
    showCPD(dbn)
    # 输出图像
    diGraphShow(dbn)

    return dbn

def evaluate(test_data, model=None):
    """evaluate the dbn model based on dataset"""

    # 推断部分
    if not model:
        with open(file_path, 'rb') as f:
            dbn = pickle.load(f)
    else:
        dbn = model

    # 输出图像
    diGraphShow(dbn)

    for test_sample in test_data:
        inferring(dbn, test_sample)


if __name__ == "__main__":
    # 设置NUMEXPR_MAX_THREADS环境变量
    os.environ["NUMEXPR_MAX_THREADS"] = "8"  # 设置为所需的线程数
    pd.options.mode.chained_assignment = None  # 默认为 'warn'
    file_path = 'trained_dbn_model_straight.pkl'

    # 设置训练和评估的类型
    # [是否训练、是否保存模型、是否评估模型]
    mode_select = {
        'notSave': {'train': True, 'save': False, 'evaluate': True},
        'evaluateOnly': {'train': False, 'save': False, 'evaluate': True},
        'updateModel': {'train': True, 'save': True, 'evaluate': True}
    }
    mode = mode_select['evaluateOnly']

    # 训练数据格式转换
    train_data1, test_data1 = trainDataTake()
    train_data2, test_data2 = trainDataTake2()

    dbn = None
    # 训练模型
    if mode['train']:
        dbn = train(train_data1, save_model=mode['save'])

    # 评估模型
    if mode['evaluate']:
        evaluate(test_data1, model=dbn)


