import os


# ---------------Dynamic Bayesian Network----------------------
TIME_SLICE = 2  # 网络的时间片

EPOCH = 10  # 针对同一数据集重复训练的次数


# 路径
LOCAL_PATH = os.path.dirname(__file__)
# 模型路径
MODEL_PATH = os.path.join(LOCAL_PATH, r"trained_dbn_model_straight.pkl")
# SinD数据统计表
META_PATH = os.path.join(LOCAL_PATH, r"data\SIND模糊场景数据\fuzzy_state_tracks_70组数据\fuzzy_state_meta.xlsx")
# SinD轨迹存储路径
NEW_PATH = os.path.join(LOCAL_PATH, r"data\SIND模糊场景数据\fuzzy_state_tracks")
# SILAB数据统计表
META_PATH_SILAB = os.path.join(LOCAL_PATH, r"data\驾驶模拟实验数据\格式化数据\total.csv")
# SILAB轨迹存储路径
NEW_PATH_SILAB = os.path.join(LOCAL_PATH, r"data\驾驶模拟实验数据\格式化数据")



# ---------------Deep Reinforcement Learning------------------

