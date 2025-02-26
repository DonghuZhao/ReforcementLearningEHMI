import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from collections import Counter

# 驾驶模拟实验仿真数据路径
simulation_data_path = r'D:\Projects\Data\实验数据'
result_path = r'D:\Projects\ReforcementLearningEHMI\ResultAnalysis'

drivers = 18
scenario_list = ['Scenario10', 'Scenario11', 'Scenario12']
# scenario_list = ['Scenario12']
strategy_list = ['A', 'B', 'C']

# # 设置显示所有行
# pd.set_option('display.max_rows', None)
#
# # 设置显示所有列
# pd.set_option('display.max_columns', None)

def read_interaction_phase(path):
    """
    .asc file columns:
    MeasurementTime,Measurement time error,ego_X,ego_Y,ego_yaw,ego_speed,ego_acc,ego_s,ego_d,
    eHMI_green_on,eHMI_red_on,HV_X,HV_Y,HV_yaw,HV_v_km/h,HV_ax,HV_ay,Straight_SteeringWheel,
    Straight_AcceleratorPedal,Straight_BrakePedal,K1.NumPad1,K1.NumPad2,time_cost[s],reward[%]
    """
    interaction_phase = pd.read_csv(path, sep=',', header=0,
                                    usecols=['ego_X', 'ego_Y', 'ego_yaw', 'ego_speed', 'ego_acc', 'ego_s', 'ego_d',
                                             'HV_X', 'HV_Y', 'HV_yaw', 'HV_v_km/h', 'HV_ax', 'HV_ay', 'time_cost[s]', 'reward[%]'])
    # 除去预热部分
    interaction_phase = interaction_phase[interaction_phase['time_cost[s]'] > 0]

    return interaction_phase

def fix_ego_speed(interaction_phase):
    """fix ego vehicle speed from trajectory (interaction phase)."""
    start_index = interaction_phase.index[0]
    end_index = np.where(interaction_phase['ego_speed'] > 0)[0][0]
    # 注意np.where返回的是索引位置（通常是行号）, 不是index
    fix_speed = interaction_phase.iloc[end_index]['ego_speed']
    interaction_phase.loc[start_index:start_index + end_index, 'ego_speed'] = fix_speed

    return interaction_phase

def calc_conflict_point(interaction_phase):
    """calculate conflict point from trajectory (interaction phase).
    Param: interaction_phase.
    return:
        min_distance: minimax distance between trajectory point.
        min_ego_index: index that ego vehicle reach minimax distance from HV.
        min_HV_index: index that HV reach minimax distance from ego vehicle.
    """
    min_distance = 1000
    min_ego_index = 0
    min_HV_index = 0
    for ego_index in interaction_phase.index:
        for HV_index in interaction_phase.index:
            ego_X = interaction_phase.loc[ego_index, 'ego_X']
            ego_Y = interaction_phase.loc[ego_index, 'ego_Y']
            HV_X = interaction_phase.loc[HV_index, 'HV_X']
            HV_Y = interaction_phase.loc[HV_index, 'HV_Y']
            distance = np.sqrt((ego_X - HV_X) ** 2 + (ego_Y - HV_Y) ** 2)
            if distance < min_distance:
                min_distance = distance
                min_ego_index = ego_index
                min_HV_index = HV_index

    return min_distance, min_ego_index, min_HV_index


def calc_pet(interaction_phase):
    """calculate PET[s] from trajectory (interaction phase)."""
    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    MAX_SPEED = 20.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = 1.
    """ Minimum reachable speed [m/s] """

    min_distance, min_ego_index, min_HV_index = calc_conflict_point(interaction_phase)

    ego_v = max(interaction_phase.loc[min_ego_index, 'ego_speed'], MIN_SPEED)
    HV_v = max(interaction_phase.loc[min_HV_index, 'HV_v_km/h'] / 3.6, MIN_SPEED)

    pet = abs(min_ego_index - min_HV_index) / 10 - LENGTH / 2 / ego_v - LENGTH / 2 / HV_v

    return pet

def calc_deltaTTCP(interaction_phase):
    """calculate deltaTTCP[s] from trajectory (interaction phase).
    start line of interaction: left vehicle(X: -25m)、 straight vehicle(X: 50m)"""

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    MAX_SPEED = 20.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = 1.
    """ Minimum reachable speed [m/s] """
    MIN_DELTATTCP = -10.
    """ Minimum deltaTTCP [s] """
    MAX_DELTATTCP = 10.
    """ Maximum deltaTTCP [s] """

    min_distance, min_ego_index, min_HV_index = calc_conflict_point(interaction_phase)
    conflict_X = (interaction_phase.loc[min_ego_index, 'ego_X'] + interaction_phase.loc[min_HV_index, 'HV_X']) / 2
    conflict_Y = (interaction_phase.loc[min_ego_index, 'ego_Y'] + interaction_phase.loc[min_HV_index, 'HV_Y']) / 2
    interaction_phase['ego_distance2CP'] = np.sqrt((conflict_X - interaction_phase['ego_X']) ** 2 + (conflict_Y - interaction_phase['ego_Y']) ** 2)
    interaction_phase['HV_distance2CP'] = np.sqrt((conflict_X - interaction_phase['HV_X']) ** 2 + (conflict_Y - interaction_phase['HV_Y']) ** 2)

    for index in interaction_phase.index:
        ego_v = interaction_phase.loc[index, 'ego_speed']
        ego_v = 5.0 if ego_v == 0 else ego_v
        ego_v = max(ego_v, MIN_SPEED)
        HV_v = max(interaction_phase.loc[index, 'HV_v_km/h'] / 3.6, MIN_SPEED)
        interaction_phase.loc[index, 'deltaTTCP'] = (interaction_phase.loc[index, 'HV_distance2CP'] / HV_v
                                                     - interaction_phase.loc[index, 'ego_distance2CP'] / ego_v)

    interaction_phase.loc[interaction_phase['deltaTTCP'] < MIN_DELTATTCP, 'deltaTTCP'] = MIN_DELTATTCP
    interaction_phase.loc[interaction_phase['deltaTTCP'] > MAX_DELTATTCP, 'deltaTTCP'] = MAX_DELTATTCP

    # 判断主车是左转车还是直行车从而确定开始的时刻
    if interaction_phase.iloc[0]['ego_X'] < -25:
        start_index = np.where(interaction_phase['ego_X'] >= -25)[0][0]
    else:
        start_index = np.where(interaction_phase['ego_X'] <= 50)[0][0]

    start_index = start_index + interaction_phase.index[0]
    # 结束时刻为某一方到达冲突点
    end_index = min(min_ego_index, min_HV_index)

    # print(interaction_phase.loc[start_index:end_index, ['HV_X', 'HV_Y', 'ego_distance2CP', 'HV_distance2CP', 'deltaTTCP']])

    return interaction_phase.loc[start_index:end_index, ['deltaTTCP']]

def calc_interaction_complete_timing(deltaTTCP):
    """calculate interaction complete timing from trajectory (interaction phase).
    Params:
        deltaTTCP: dataframe of deltaTTCP between two vehicle.
        scenario: scenario that in which the interaction happens.
    Return:
        timing: time index that the interaction between two vehicle is completed, meaning safe.
    Principle:
        when the deltaTTCP is bigger than 2.0s, the collision is solved."""
    THRESHOLD = 2.0

    # 先根据最后一帧判断通行次序，再查找所有deltaTTCP < 2.0s的index，返回最后一个
    pass_seq = deltaTTCP.iloc[-1]['deltaTTCP']
    if pass_seq > 0:
        collision_index = np.where(deltaTTCP['deltaTTCP'] < THRESHOLD)[0]
        return collision_index[-1] if len(collision_index) > 0 else 0
    else:
        collision_index = np.where(deltaTTCP['deltaTTCP'] > - THRESHOLD)[0]
        return collision_index[-1] if len(collision_index) > 0 else 0

def calc_interaction_times(deltaTTCP):
    THRESHOLD = 1.0
    qujian = deltaTTCP['deltaTTCP'].apply(lambda x: 1 if x > THRESHOLD else (-1 if x < -THRESHOLD else 0))
    qujianchange = abs(qujian.diff())
    times = int(qujianchange.sum())
    return times

def plot_trajectory(df):
    plt.figure()
    plt.plot(df['ego_X'], df['ego_Y'], 'r')
    plt.plot(df['HV_X'], df['HV_Y'], 'b')
    plt.show()

def plot_velocity(df):
    plt.figure()
    plt.plot(df['ego_speed'], 'r')
    plt.plot(df['HV_v_km/h'] / 3.6, 'b')
    plt.show()


time_cost = {
    'Scenario10': {
        'A': [],
        'B': [],
        'C': []
    },
    'Scenario11': {
        'A': [],
        'B': [],
        'C': []
    },
    'Scenario12': {
        'A': [],
        'B': [],
        'C': []
    }
}

pet = copy.deepcopy(time_cost)
collision = copy.deepcopy(time_cost)
interaction_complete = copy.deepcopy(time_cost)
df_TTCP = pd.DataFrame(columns=['scenario', 'strategy', 'index', 'deltaTTCP'])
disobey = copy.deepcopy(time_cost)
times = copy.deepcopy(time_cost)

# -----------指标计算-----------------
for scenario in scenario_list:
    print(scenario, "calculating...")
    for strategy in strategy_list:
        for driver in range(1, drivers + 1):
            # 数据路径
            path = os.path.join(simulation_data_path, '驾驶人{}'.format(driver), '{}_{}.asc'.format(scenario, strategy),)
            interaction_phase = read_interaction_phase(path)
            interaction_phase = fix_ego_speed(interaction_phase)
            # Part1 time cost
            # single_time_cost = interaction_phase['time_cost[s]'].max()
            # time_cost[scenario][strategy].append(single_time_cost)
            # Part2 collision
            final_reward = interaction_phase.iloc[-1]['reward[%]']
            if final_reward == 0:
                collision[scenario][strategy].append(1)
                continue
            # Part3 deltaTTCP
            deltaTTCP = calc_deltaTTCP(interaction_phase)
            if (scenario == 'Scenario10' and deltaTTCP.iloc[-1]['deltaTTCP'] > 0 or
                scenario == 'Scenario11' and deltaTTCP.iloc[-1]['deltaTTCP'] < 0 or
                scenario == 'Scenario12' and deltaTTCP.iloc[-1]['deltaTTCP'] > 0):
                disobey[scenario][strategy].append(1)
                continue
            times[scenario][strategy].append(calc_interaction_times(deltaTTCP))
            interaction_complete[scenario][strategy].append(calc_interaction_complete_timing(deltaTTCP))
            deltaTTCP['scenario'] = scenario
            deltaTTCP['strategy'] = strategy
            df_TTCP = pd.merge(df_TTCP, deltaTTCP.reset_index(drop=True).reset_index(),
                               how='outer', on=['scenario', 'strategy', 'index', 'deltaTTCP'])
            # Part4 PET
            pet_ = calc_pet(interaction_phase)
            pet[scenario][strategy].append(pet_)
# print("time_cost:", time_cost)
# print("pet:", pet)
# 存储为pkl文件
with open(os.path.join(result_path, 'pet.pkl'), 'wb') as f:
    pickle.dump(pet, f)

with open(os.path.join(result_path, 'time_cost.pkl'), 'wb') as f:
    pickle.dump(time_cost, f)

with open(os.path.join(result_path, 'interaction_complete.pkl'), 'wb') as f:
    pickle.dump(interaction_complete, f)

# ---------------读取pkl文件-----------
with open(os.path.join(result_path, 'time_cost.pkl'), 'rb') as f:
    time_cost = pickle.load(f)

with open(os.path.join(result_path, 'pet.pkl'), 'rb') as f:
    pet = pickle.load(f)

with open(os.path.join(result_path, 'interaction_complete.pkl'), 'rb') as f:
    interaction_complete = pickle.load(f)

print("time_cost:", time_cost)
print("pet:", pet)
print("collision:", collision)
print("interaction_complete:", interaction_complete)

# Time Cost 绘制箱型图
for scenario in scenario_list:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=time_cost[scenario])
    plt.title('Box Plot Comparison of Three Strategies', fontsize=20)
    plt.xlabel('Strategy', fontsize=20)
    plt.ylabel('TimeCost[s]', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylim(0, 25)
    plt.show()

# PET 绘制箱型图
for scenario in scenario_list:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pet[scenario])
    plt.title('Box Plot Comparison of Three Strategies', fontsize=20)
    plt.xlabel('Strategy', fontsize=20)
    plt.ylabel('PET[s]', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    # plt.ylim(0, 25)
    plt.show()

# Time Cost统计信息
print("Time Cost Summary:")
for scenario in scenario_list:
    for strategy in strategy_list:
        print(scenario, strategy, np.mean(time_cost[scenario][strategy][:5]), np.std(time_cost[scenario][strategy][:5]))

# PET 统计信息
print("PET Summary:")
for scenario in scenario_list:
    for strategy in strategy_list:
        print(scenario, strategy, np.mean(pet[scenario][strategy][:5]), np.std(pet[scenario][strategy][:5]))

df_TTCP.to_csv(os.path.join(result_path, 'deltaTTCP.csv'), index=False)
df_TTCP = pd.read_csv(os.path.join(result_path, 'deltaTTCP.csv'))

# 冲突消除
# Scenario10
sns.lineplot(data=df_TTCP[df_TTCP['scenario']=='Scenario10'], x='index', y='deltaTTCP', hue='strategy', errorbar=('ci', 95))
# plt.xlim([40, 125])
plt.show()

sns.lineplot(data=df_TTCP[df_TTCP['scenario']=='Scenario11'], x='index', y='deltaTTCP', hue='strategy', errorbar=('ci', 95))
# plt.xlim([20, 60])
plt.show()

sns.lineplot(data=df_TTCP[df_TTCP['scenario']=='Scenario12'], x='index', y='deltaTTCP', hue='strategy', errorbar=('ci', 95))
# plt.xlim([30, 90])
plt.show()

print("interaction_complete:", interaction_complete)
# interaction_complete 绘制箱型图
for scenario in scenario_list:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=interaction_complete[scenario])
    plt.title('Box Plot Comparison of Three Strategies', fontsize=20)
    plt.xlabel('Strategy', fontsize=20)
    plt.ylabel('Timing[s]', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.show()

print("disobey:", disobey)
print("times:", times)
for scenario in scenario_list:
    for strategy in strategy_list:
        print(scenario, strategy, Counter(times[scenario][strategy]))