import os
import numpy as np
# ---------------Dynamic Bayesian Network----------------------
TIME_SLICE = 2  # 网络的时间片
EPOCH = 10  # 针对同一数据集重复训练的次数
STEP = 5  # 轨迹步长单位/0.1s
HISTORY_LENGTH = 20 # 用于推理的历史轨迹长度单位/0.1s

# 识别对象
OBJECTS = {
    0: 'Straight',
    1: 'Left'
}
TARGET_OBJECT = OBJECTS[0]

# 连续变量的离散化阈值
I_values = [0, 1]
Dp_values = list(np.arange(-20, 21))
Dv_values = list(np.arange(-10, 11))
V_values = list(np.arange(0, 11))
D_values = list(np.arange(0, 41, 5))

# 路径
LOCAL_PATH = os.path.dirname(__file__)
# 模型路径
MODEL_PATH_STRAIGHT = os.path.join(LOCAL_PATH, r"trained_dbn_model_straight.pkl")   # 直行车意图识别模型
MODEL_PATH_LEFT = os.path.join(LOCAL_PATH, r"trained_dbn_model.pkl")    # 左转车意图识别模型
# SinD数据统计表
META_PATH = os.path.join(LOCAL_PATH, r"data\SIND模糊场景数据\fuzzy_state_tracks_70组数据\fuzzy_state_meta.xlsx")
# SinD轨迹存储路径
TRACK_PATH_SIND = os.path.join(LOCAL_PATH, r"data\SIND模糊场景数据\fuzzy_state_tracks")
# SILAB数据统计表
META_PATH_SILAB = os.path.join(LOCAL_PATH, r"data\驾驶模拟实验数据\格式化数据\total.csv")
# SILAB轨迹存储路径
TRACK_PATH_SILAB = os.path.join(LOCAL_PATH, r"data\驾驶模拟实验数据\格式化数据")



# ---------------Deep Reinforcement Learning------------------

