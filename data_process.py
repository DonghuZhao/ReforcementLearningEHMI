# -*- coding: utf-8 -*-
%config InlineBackend.figure_format='svg'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def SilabXY_to_EditXY(x,y,vx,vy,ax,ay,yaw):
    WIDTH = 266.7007
    editx = WIDTH - x
    edity = y + 16.3609
    edityaw = 3.1415926 - yaw
    editvx = (vx*math.cos(yaw) - vy*math.sin(yaw)) * (-1)
    editvy = vx*math.sin(yaw) + vy*math.cos(yaw)
    editax = (ax*math.cos(yaw) - ay*math.sin(yaw)) * (-1)
    editay = ax*math.sin(yaw) + ay*math.cos(yaw)
    return [editx,edity,editvx,editvy,editax,editay]


intention_X = []
# StopLine = 33.46
StopLine = 266.7007 - 33.46
convey_cost = []

start = -25

pd.options.mode.chained_assignment = None  # 默认为 'warn'
path = r"驾驶数据\格式化数据"

for driver in range(0,6):
    driver_path = os.path.join(path,"driver_{}".format(driver + 1))
    if os.path.exists(driver_path):
        pass
    else: 
        os.makedirs(driver_path)
    
    for scenerio in range(18):
        scenerio_path = os.path.join(driver_path, "scenerio_{}".format(scenerio + 1))
        if os.path.exists(scenerio_path):
            pass
        else:
            os.makedirs(scenerio_path)
            
        file_path = f"C:\\Users\\ZDH\\OneDrive\\交互行为\\SILAB第三次实验\\驾驶数据\\驾驶员{driver+1}\\DataFile."+ str( scenerio + 1 ) + ".asc"
        data_all = pd.read_csv(file_path)
        print("read",scenerio)
        
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
        
        # 坐标转换
        data_single["Straight_X"] = 266.7007 - data_single["Straight_X"]
        data_single["Straight_Y"] = data_single["Straight_Y"] + 16.3609
        data_single["Straight_yaw"] = 3.1415926 - data_single["Straight_yaw"]
        
        start_index = np.where(data_single["Left_X"] > start)[0][0]
        # print("start_index:", start_index)
        try:
            end_index = np.where(data_single["Straight_X"] < 0)[0][0]
        except:
            end_index = data_single.index[-1]
        # print("end_index:", end_index)
        
        data_single = data_single.loc[start_index:end_index]
        data_single["Straight_Vx"] = data_single["Straight_X"].diff().bfill() * 10
        data_single["Straight_ax"] = data_single["Straight_Vx"].diff().bfill() * 10
        data_single["Straight_Vy"] = data_single["Straight_Y"].diff().bfill() * 10
        data_single["Straight_ay"] = data_single["Straight_Vy"].diff().bfill() * 10
        data_single["Left_Vx"] = data_single["Left_X"].diff().bfill() * 10
        data_single["Left_ax"] = data_single["Left_Vx"].diff().bfill() * 10
        data_single["Left_Vy"] = data_single["Left_Y"].diff().bfill() * 10
        data_single["Left_ay"] = data_single["Left_Vy"].diff().bfill() * 10
        
        data_single["d_x"] = data_single["Left_X"] - data_single["Straight_X"]
        data_single["d_y"] = data_single["Left_Y"] - data_single["Straight_Y"]
        data_single["distance"] = np.sqrt((data_single["Left_X"] - data_single["Straight_X"]) ** 2 + (data_single["Straight_Y"] - data_single["Left_Y"]) ** 2)
        data_single["l_dp"] = data_single["Left_X"] + 4.36
        data_single["Dv"] = abs(data_single["Left_Vx"]) - abs(data_single["Straight_Vx"])
        data_single["s_dp"] = data_single["Straight_X"] - 33.46
        outputs = ["Left_X", "Left_Y", "Left_yaw", "Left_Vx", "Left_Vy", "Left_ax", "Left_ay",
                    "Straight_X", "Straight_Y", "Straight_yaw", "Straight_Vx", "Straight_Vy", "Straight_ax", "Straight_ay",
                    "d_x", "d_y", "distance", "l_dp", "Dv", "s_dp"]
        data_single = data_single[outputs]
        outputs = ["l_x", "l_y", "l_h", "l_vx", "l_vy", "l_ax", "L_ay",
                    "s_x", "s_y", "s_h", "s_vx", "s_vy", "s_ax", "s_ay",
                    "d_x", "d_y", "distance", "l_dp", "Dv", "s_dp"]
        data_single.columns = outputs
        data_single.to_csv(os.path.join(scenerio_path, "both.csv"), index = None)
        
#         try:
#             conflict = np.where(data_single["Straight_X"] - data_single["Left_X"] < 0)[0][0]
#             if data_single.loc[conflict, "Left_Y"] > data_single.loc[conflict, "Straight_Y"]:
#                 ego_seq = 1
#             else:
#                 ego_seq = 0
#         except:
#             ego_seq = "None"


# # In[HV_intention]
#         NumPad1 = np.where(data_single["K1.NumPad1"]!=0)[0]
#         NumPad2 = np.where(data_single["K1.NumPad2"]!=0)[0]
#         if NumPad1.any():
#             intention_recognition = NumPad1[0]
#             intention = "PREEMPT"
#         else:
#             intention_recognition = NumPad2[0]
#             intention = "YILED"
        
        
#         intention_recognition_dist = data_single.loc[intention_recognition,"Straight_X"] - StopLine
        
# # In[AV_intention]
#         AV_intention = data_single["Straight_longterm_decision_flag"].unique()[1]
        
        
#         intention_X.append([driver + 1, scenerio + 1, intention, AV_intention, ego_seq, intention_recognition - start_index])
        
# intention_X = pd.DataFrame(intention_X, columns = ["driver_id", "scenerio_id", "HV_intention","AV_intention","ego_seq", "decision_index"])
# intention_X.to_csv("驾驶数据/格式化数据/total.csv", index = None)
 


