
import pickle
import pandas as pd
import os
import numpy as np

meta_path = r"C:\Users\ZDH\OneDrive - tongji.edu.cn\硕士\代码\数据\SIND模糊场景数据\fuzzy_state_tracks_70组数据\fuzzy_state_meta.xlsx"
file_path = r"C:\Users\ZDH\OneDrive - tongji.edu.cn\硕士\代码\数据\SIND模糊场景数据\fuzzy_state_tracks_70组数据\hv_smoothed_tracks_total.xlsx"
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

# 分开左转和直行
def split_track(o_data):
    ids = o_data["track_id"].unique()
    left_track = o_data[o_data["track_id"]==ids[0]]
    left_track = track_from_data(left_track)
    straight_track = o_data[o_data["track_id"]==ids[1]]
    straight_track = track_from_data(straight_track)
    return left_track, straight_track
# 数据格式转换
def track_from_data(data):
    track = Track_info()
    track.timestep = data["timestamp_ms"].values.tolist()
    track.position = data[["x","y"]].values.tolist()
    track.velocity = data[["vx","vy"]].values.tolist()
    track.heading = data["yaw_rad"].values.tolist()
    return track

def _calc_interaction_metrix(left_track, straight_track):
    res = pd.DataFrame(columns=["l_x","l_y","l_h","l_vx","l_vy","l_ax","l_ay",
                                "s_x","s_y","s_h","s_vx","s_vy","s_ax","s_ay",
                                "d_x","d_y","distance"])
    res[["l_x","l_y"]] = left_track.position
    res["l_h"] = left_track.heading
    res[["l_vx","l_vy"]] = left_track.velocity
    res[["l_ax","l_ay"]] = res[["l_vx","l_vy"]].diff()
    res[["s_x","s_y"]] = straight_track.position
    res["s_h"] = straight_track.heading
    res[["s_vx","s_vy"]] = straight_track.velocity
    res[["s_ax","s_ay"]] = res[["s_vx","s_vy"]].diff()
    res["d_x"] = res["l_x"] - res["s_x"]
    res["d_y"] = res["l_y"] - res["s_y"]
    res["distance"] = np.sqrt(res["d_x"] * res["d_x"] + res["d_y"] * res["d_y"])
    res["l_dp"] = res["l_x"] + 4.36
    res["Dv"] = abs(res["l_vx"]) - abs(res["s_vx"])
    res["s_dp"] = res["s_x"] - 33.46
    return res

# 提取需要的交互对
meta = pd.read_excel(meta_path, sheet_name="Sheet1", header=0, skiprows=[1])
# meta = meta[meta["ego_entrance"] == 1]
# meta = meta[meta["sig_state"] != 0 ]

# 轨迹存储路径
new_path = r"C:\Users\ZDH\OneDrive - tongji.edu.cn\硕士\代码\数据\SIND模糊场景数据\fuzzy_state_tracks"

for index,sample in meta.iterrows():
    _id = int(sample["id"])
    print("id:",_id)
    folder = os.path.join(new_path,str(_id))
    if not os.path.exists(folder):
        os.mkdir(folder)
    o_data = pd.read_excel(file_path, sheet_name=str(_id), header = 0)
    left_track, straight_track = split_track(o_data)
# # 左转车轨迹信息存储
#     _pkl_left = os.path.join(folder,"left.data")
#     with open(_pkl_left, 'wb') as filehandle:
#         pickle.dump(left_track, filehandle)
# # 直行车轨迹信息存储
#     _pkl_straight = os.path.join(folder,"straight.data")
#     with open(_pkl_straight, 'wb') as filehandle:
#         pickle.dump(straight_track, filehandle)

# 训练数据格式
    res = _calc_interaction_metrix(left_track, straight_track)
    _res_path = os.path.join(folder, "both.csv")
    res.to_csv(_res_path, index=False)






