import pandas as pd
import numpy as np
import random
from config import *

def loadSinDData(targetObject='Straight'):
    """
    load SinD data
    """
    col_indexes = [10, 16, 18, 19]
    if targetObject == 'Left':
        col_indexes = [3, 16, 17, 18]
    # 训练数据
    meta_path = META_PATH
    # 提取需要的交互对
    meta = pd.read_excel(meta_path, sheet_name="Sheet1", header=0, skiprows=[1])
    meta = meta[meta["ego_entrance"] == 1] # 左转车进口道为西进口道
    meta = meta[meta["sig_state"] != 0]    # 信号灯为绿灯
    # 轨迹存储路径
    track_path = TRACK_PATH_SIND

    train_data_combined = pd.DataFrame()
    test_data_combined = pd.DataFrame()
    test_set = []
    # 将分散的交互对合并成一个训练数据集
    for index, sample in meta.iterrows():
        # if sample["pass_seq"] == 1:
        #     continue
        _id = int(sample["id"])
        print(_id, "read", "---result:", sample["pass_seq"])
        folder = os.path.join(track_path, str(_id))

        data_file = os.path.join(folder, "both.csv")
        with open(data_file, 'rb') as filehandle:
            train_data = pd.read_csv(data_file, header=0, usecols=col_indexes)

        # 按照一定步长对数据进行精简以
        train_data = train_data[train_data.index % STEP == 0]

        # 数据编辑
        train_data.rename(columns={"Dv": "Dv", "l_dp": "Dp", "distance": "D",
                                   "l_vx": "V", "s_dp": "Dp", "s_vx": "V"}, inplace=True)

        # 修正数据
        train_data['V'] = abs(train_data['V'])

        train_data = dataDiscretization(train_data)
        # if pass_seq == 1 左转先行， 对应直行车的 Yield_probability
        # 这里统一用intention == 1来表示让行
        train_data["I"] = sample["pass_seq"] if targetObject == 'Straight' else 1 - sample["pass_seq"]
        train_data = train_data.astype(int)
        # print(train_data)
        # 筛选测试集
        if _id in [35, 17]:   # 35 & 62, 17
            # test_data_combined = pd.concat([test_data_combined, train_data])
            test_set.append(train_data)
        else:
            train_data = dataStructTransform(train_data)
            train_data_combined = pd.concat([train_data_combined, train_data])

    print("train dataset:\n", train_data_combined)
    print("test dataset:\n", test_set)
    return train_data_combined, test_set

def loadSilabData(targetObject='Straight'):
    """
    load Silab Data
    """
    col_indexes = [10, 16, 18, 19]
    if targetObject == 'Left':
        col_indexes = [3, 16, 17, 18]
    # 训练数据
    meta_path = META_PATH_SILAB
    # 提取需要的交互对
    meta = pd.read_csv(meta_path,header=0)

    # 轨迹存储路径
    track_path = TRACK_PATH_SILAB

    train_data_combined = pd.DataFrame()
    test_data_combined = pd.DataFrame()
    test_set = []
    # 将分散的交互对合并成一个训练数据集
    for index, sample in meta.iterrows():
        driver = int(sample["driver_id"])
        scenerio = int(sample["scenerio_id"])
        print("driver{}scenerio{}".format(driver, scenerio), "read", "---result:", sample["ego_seq"])
        folder = os.path.join(track_path,"driver_{}".format(driver),"scenerio_{}".format(scenerio))

        data_file = os.path.join(folder, "both.csv")
        with open(data_file, 'rb') as filehandle:
            train_data = pd.read_csv(data_file, header=0, usecols=col_indexes)

        # 按照一定步长对数据进行精简以
        train_data = train_data[train_data.index % STEP == 0]

        # 数据编辑, l和s不会同时出现
        train_data.rename(columns={"Dv": "Dv", "l_dp": "Dp", "distance": "D",
                                   "l_vx": "V", "s_dp": "Dp", "s_vx": "V"}, inplace=True)

        # 修正数据
        train_data['V'] = abs(train_data['V'])

        train_data = dataDiscretization(train_data)
        # if pass_seq == 1 左转先行， 对应直行车的 Yield_probability
        train_data["I"] = sample["ego_seq"] if targetObject == 'Straight' else 1 - sample["ego_seq"]
        train_data = train_data.astype(int)

        # 区分训练集和测试集
        if random.random() < 0.2:
            test_set.append(train_data)
        else:
            train_data = dataStructTransform(train_data)
            train_data_combined = pd.concat([train_data_combined, train_data])

    print("train dataset:\n")
    print(train_data_combined)
    print("test dataset:\n")
    print(test_set)
    return train_data_combined, test_set

def dataDiscretization(train_data):
    """transform continuous variables to discrete variables"""
    train_data["Dp"] = train_data["Dp"].apply(continousToDiscrete, criteria=Dp_values)
    train_data["Dv"] = train_data["Dv"].apply(continousToDiscrete, criteria=Dv_values)
    train_data["V"] = train_data["V"].apply(continousToDiscrete, criteria=V_values)
    train_data["D"] = train_data["D"].apply(continousToDiscrete, criteria=D_values)
    return train_data

def dataStructTransform(data):
    """
    将时序序列数据转化为可用于DBN模型训练的两个时间片数据
    例：[x0,x1,x2,x3,x4]转化为[(x0, x1), (x1, x2), (x2, x3), (x3, x4)]
    """
    # 将 DataFrame 转换为一行并添加索引到列名
    _, attributesNum = data.shape
    # Step1：行复制
    data = pd.DataFrame(np.repeat(data.values, repeats=2, axis=0))[1:-1]
    # Step2：修改列名
    merged_data_columns = []
    for t in range(TIME_SLICE):
        merged_data_columns.extend([('V', t), ('D', t), ('Dp', t), ('Dv', t), ('I', t)])
    # Step3：重组时序数据
    data_structed = pd.DataFrame(data.values.reshape(-1, attributesNum * TIME_SLICE), columns=merged_data_columns)
    return data_structed

def continousToDiscrete(variable, criteria):
    """
    连续变量离散化
    """
    if len(criteria) == 0:
        raise ValueError("criteria is empty")
    for i in range(len(criteria)):
        if variable < criteria[i]:
            break
    return i

def testDataTake():
    testData_set = []
    for driver in range(4, 5):
        for scenerio in range(10, 15):
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
            data_single['Dv'] = data_single['l_vx'] - data_single['s_vx']
            # 数据编辑, l和s不会同时出现
            data_single.rename(columns={"Dv": "Dv", "l_dp": "Dp", "distance": "D",
                                       "l_vx": "V", "s_dp": "Dp", "s_vx": "V"}, inplace=True)
            testData_set.append(dataDiscretization(data_single[["s_x", "s_vx", "D", "l_vx"]]))
    return testData_set