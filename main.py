import torch
from PPO import Agent
import numpy as np
import utils
import time

from map.config import *

# 1. load model
agent = Agent(agent_path=r'.\model\agent.json')
agent.loadNetParas()
features_range = agent.features_range

# "features": ["x", "y", "speed", "heading","lat_off", "ang_off"],



# Quest:
"""
    主车信息
    ego_vehicle:{'x': 1, 'y':1, 'speed':1, 'heading':1, 'lat_off':1, 'ang_off':1, }
    环境车信息（车辆ID：车辆信息）
    vehicles:{
                0: {'x': 1, 'y':1, 'speed':1, 'heading':1, 'lat_off':1, 'ang_off':1, },
                ...
            }
"""
class TrajPlanQuest:
    def __init__(self):
        self.s = 0
        self.d = 0
        self.position = np.array([0,0])
        self.speed = 0
        self.heading = 0
        self.ob = np.array([0,0])
        self.ob_v = 0
        self.ob_yaw = 0

def getStateFromQuest(quest):
    ego_vehicle = {}
    vehicles = {}

    ego_vehicle['x'] = quest.position[0]
    ego_vehicle['y'] = quest.position[1]
    ego_vehicle['speed'] = quest.speed
    ego_vehicle['heading'] = quest.heading
    ego_vehicle['lat_off'] = quest.d
    ego_vehicle['ang_off'] = quest.heading - csp.calc_yaw(quest.s)

    obj = {}
    obj['x'] = quest.ob[0]
    obj['y'] = quest.ob[1]
    obj['speed'] = np.ob_v
    obj['heading'] = np.ob_yaw

    vehicles[0] = obj

    return ego_vehicle, vehicles

def normalize_observation(s, clip=True):
    """normalize observation to [0,1]"""
    for feature, f_range in features_range:
        if feature in s:
            s[feature] = utils.lmap(s[feature], [f_range[0], f_range[1]], [-1, 1])
            if clip:
                s[feature] = np.clip(s[feature], -1, 1)
    return s

def drlControl(quest):
    """main function to output control value from state"""
    ego_vehicle, vehicles = getStateFromQuest(quest)

    # 将参照物的状态处理为雷达信息
    lidar_obs = process_radar_information(ego_vehicle, vehicles)

    s = np.concatenate(ego_vehicle.values(), lidar_obs[0])

    s = normalize_observation(s)
    # s = np.array([0,0,0,0,0,0,0,0,0])
    s = torch.tensor(s, dtype=torch.float)
    a = agent.choose_action(s)

    EHMI = 'None'
    if len(a) == 3:
        EHMI = agent.translateEHMI(a[2])
    controls = {
        'acceleration': a[0],      # m/s2
        'steering': a[1],      # m/s2
        'EHMI': EHMI,
    }
    print(f"controls:{controls}")

    return controls

if __name__ == '__main__':
    print()