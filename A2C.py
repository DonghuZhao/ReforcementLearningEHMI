import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import trange
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../InteractionPlanning/python")))
from frenet_optimal_trajectory import *
import shapely

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA=0.99
LR=0.001
MAX_EPS = 200000
GRAD_CLIP_NORM = 5
TIME = time.strftime("%Y%m%d%H%M%S",time.localtime())

class FrenetPlanner():
    def __init__(self):
        self.refline, self.csp = self.LeftRefLineBuild()
    def LeftRefLineBuild(self):
        # 构造参考线
        P_ex_extend = np.array([-50, -14.31])
        P_ex = np.array([-26.44, -14.31])
        P_en = np.array([14.74, -43.67])
        P0 = np.array([-4.39, -14.54])
        P3 = np.array([14.85, -35.40])
        P1_x, P1_y = construct_line(P0[0], P0[1], P_ex[0], P_ex[1], x=10)
        P1 = np.array([P1_x, P1_y])
        P2_x, P2_y = construct_line(P3[0], P3[1], P_en[0], P_en[1], y=20)
        P2 = np.array([P2_x, P2_y])
        t = np.linspace(0, 1, 100)
        B = np.zeros((100, 2))
        for i in range(100):
            B[i] = (1 - t[i]) ** 3 * P0 + 3 * t[i] * (1 - t[i]) ** 2 * P1 + 3 * t[i] ** 2 * (1 - t[i]) * P2 + t[
                i] ** 3 * P3

        reference_line = np.vstack((P_ex_extend, P_ex))
        reference_line = np.vstack((reference_line, B))
        reference_line = np.vstack((reference_line, P_en))

        wx = reference_line[:, 0]
        wy = reference_line[:, 1]
        tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
        return reference_line, csp

    def EditXY_to_Frenet(self, position, velocity):
        print(position)
        traj_points = MultiPoint(np.array([position]))
        traj_s, traj_l = self.calc_S_L(traj_points, LineString(self.refline))
        yaw = self.csp.calc_yaw(traj_s[0])
        s_d = velocity[0] * np.cos(yaw) + velocity[1] * np.sin(yaw)
        d_d = velocity[0] * np.sin(yaw) + velocity[1] * np.cos(yaw)
        return traj_s[0], traj_l[0], s_d, d_d

    def calc_S_L(self, traj_points, midline_lane):
        # 计算S和L
        traj_l = []
        traj_s = []

        for idx, traj_point in enumerate(traj_points.geoms):
            # 轨迹点在中心线上的投影
            nearest_mid_point = midline_lane.interpolate(midline_lane.project(traj_point))
            print("nearest_mid_point:", nearest_mid_point)
            trans_xs = (nearest_mid_point.x - traj_point.x) * 2
            trans_ys = (nearest_mid_point.y - traj_point.y) * 2
            trans_point = shapely.affinity.translate(traj_point, trans_xs, trans_ys)
            vertical_line = LineString([traj_point, trans_point])
            mid_point = vertical_line.intersection(midline_lane)
            l_dis = traj_point.distance(mid_point)
            # 判断正负
            if nearest_mid_point.y < traj_point.y:
                l_dis = l_dis
            else:
                l_dis = -l_dis

            polygon_split = shapely.ops.split(midline_lane, vertical_line)
            midline_lane_s = polygon_split.geoms[0]
            traj_s.append(midline_lane_s.length)
            traj_l.append(l_dis)

        return traj_s, traj_l

    def frenet_optimal_planner(self,s):
        position = [s[1],s[2]]
        velocity = [s[3],s[4]]
        s, d, s_d, d_d = self.EditXY_to_Frenet(position, velocity)
        bestpath, fplist_all= frenet_optimal_planning(self.csp, s, s_d, 0, d, d_d, 0, [0,0], [0,0])

        return bestpath, fplist_all



class ActorNet(nn.Module):
    def __init__(self, state_size,action_size,hidden_dim=[256,128,64]):
        super().__init__()

        self.hidden_1 = nn.Linear(state_size, hidden_dim[0])
        self.hidden_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.hidden_3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.output = nn.Linear(hidden_dim[2], action_size)

    def forward(self, s):
        outs = F.relu(self.hidden_1(s))
        outs = F.relu(self.hidden_2(outs))
        outs = F.relu(self.hidden_3(outs))
        actions_mu = self.output(outs)
        return actions_mu

class ValueNet(nn.Module):
    def __init__(self,state_size, hidden_dim=[256,128,64]):
        super().__init__()

        self.hidden_1 = nn.Linear(state_size, hidden_dim[0])
        self.hidden_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.hidden_3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.output = nn.Linear(hidden_dim[2], 1)

    def forward(self, s):
        outs = F.relu(self.hidden_1(s))
        outs = F.relu(self.hidden_2(outs))
        outs = F.relu(self.hidden_3(outs))
        value = self.output(outs)
        return value

class A2C(nn.Module):
    def __init__(self, env):
        super().__init__() 
        self.env = env
        self.state_size = int(self.env.observation_space.shape[0] * self.env.observation_space.shape[1])
        self.action_size = self.env.action_space.shape[0] + 1  # 增加EHMI的维度
        self.actor_func = ActorNet(self.state_size,self.action_size).to(device)
        self.value_func = ValueNet(self.state_size).to(device)
        self.gamma = GAMMA
        self.reward_records = []
        self.opt1 = torch.optim.AdamW(self.value_func.parameters(), lr=LR)
        self.opt2 = torch.optim.AdamW(self.actor_func.parameters(), lr=LR)
        self.best_score = None
        self.writer = SummaryWriter(comment='-'+"Obstacle_avoidance" +TIME)
        self.frenetplanner = FrenetPlanner()

    # pick up action with above distribution policy_pi
    # TODO 规划决策使用一个Frenet轨迹规划器
    def pick_motion(self,s):
        bestpath, fplist_all = self.frenetplanner.frenet_optimal_planner(s)
        if bestpath:
            return bestpath.x[0], bestpath.y[0]
        else:
            return fplist_all[0].x[0], fplist_all[0].y[0]

    def pick_sample(self,s):
        with torch.no_grad():
            # observation降维
            s = s.reshape(-1,self.state_size)
            s = s.squeeze(0)
            x, y = self.pick_motion(s)
            print("next_motion:", x, y)
            # 增加维度
            s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
            # Get logits from state
            logits = self.actor_func(s_batch)
            # logits 的形状是 (1, 1, 3)，前两个是连续变量的均值，最后一个是布尔变量的logit
            logits = logits.squeeze(dim=0)

            # 提取连续变量的均值和布尔变量的logit
            means = logits[:2]
            bool_logit = logits[2]
            # 设定标准差，可以是固定值，也可以是通过网络输出
            std = torch.tensor([0.1, 0.1], dtype=torch.float).to(device)
            # 从高斯分布中采样连续变量
            continuous_actions = torch.normal(means, std)
            # 使用sigmoid函数将布尔变量的logit转换为概率
            bool_prob = torch.sigmoid(bool_logit)

            # 从伯努利分布中采样布尔变量
            bool_action = torch.bernoulli(bool_prob)

            # 将连续变量和布尔变量组合
            a = torch.cat([continuous_actions, bool_action.unsqueeze(0)])
            # Return
            return a.tolist()
    def train(self, path):
        self.load(path)
        translate_EHMI = {0: 'Y', 1: 'N'}
        for i in trange(MAX_EPS):
            #
            # Run episode till done
            #
            if i != 0 and i % 100 == 0:
                self.save(path)
            done = False
            states = []
            actions = []
            rewards = []
            s= self.env.reset()[0]
            while not done:
                states.append(s.reshape(1,self.state_size).tolist())
                a = self.pick_sample(s)
                # 额外一步
                self.env.unwrapped.EHMI = translate_EHMI[a[2]]
                # 常规步骤
                s, r, done, truncated, info = self.env.step(a[:2])
                # print("state:", s)
                # 是否显示Highway的可视化界面
                self.env.render()
                # actions.append(np.expand_dims(a, axis=0))
                actions.append(a)
                rewards.append(r)
            # if done:
            #     print("done:", done)
                
            #
            # Get cumulative rewards
            #
            cum_rewards = np.zeros_like(rewards)
            reward_len = len(rewards)
            for j in reversed(range(reward_len)):
                cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*self.gamma if j+1 < reward_len else 0)
            cum_rewards = np.expand_dims(cum_rewards, axis=1)

            #saving the best model 
            self.reward_records.append(sum(rewards))
            if len(self.reward_records)>1 : 
                self.best_score = np.max(np.array(self.reward_records[:-1]))
            

                if self.best_score < self.reward_records[-1]:
                    torch.save(self.actor_func.state_dict(), f"bestmodel/a3c_value_model_score{self.reward_records[-1]}.pth")
                    torch.save(self.value_func.state_dict(), f"bestmodel/a3c_policy_model_score{self.reward_records[-1]}.pth")

            self.writer.add_scalar("Avg_Rewards_100",np.mean(np.array(self.reward_records[-100:])),i)
            # Train (optimize parameters)
            #
            # Optimize value loss (Critic)
            self.opt1.zero_grad()
            states = torch.tensor(states, dtype=torch.float).to(device)
            cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
            values = self.value_func(states)
            values = values.squeeze(dim=1)
            vf_loss = F.mse_loss(values,cum_rewards,reduction="none")
            self.writer.add_scalar("Critic_loss",vf_loss.mean(),i)
            vf_loss.mean().backward()
            #gradient cliping
            torch.nn.utils.clip_grad_norm_(self.value_func.parameters(),GRAD_CLIP_NORM)
            self.opt1.step()

            # Optimize policy loss (Actor)
            with torch.no_grad():
                values = self.value_func(states)
            self.opt2.zero_grad()
            # actions 是实际采取的行动
            actions = torch.tensor(actions, dtype=torch.float32).to(device)
            advantages = cum_rewards - values
            # actions_mu 是模型输出的行动
            actions_mu = self.actor_func(states)
            # loss 计算
            # 连续动作部分
            actions_mu = torch.squeeze(actions_mu, dim=1)
            mu = actions_mu[:, :2]
            std = torch.tensor([[0.1, 0.1] for _ in range(len(mu))])
            # Create normal distribution and compute log probabilities
            dist = torch.distributions.Normal(mu, std)
            log_probs_continuous = dist.log_prob(actions[:, :2]).sum(-1)  # Summing log probabilities across action dimensions

            # Create Bernoulli distribution for the boolean action and compute log probabilities
            # 使用sigmoid函数将布尔变量的logit转换为概率
            bool_prob = torch.sigmoid(actions_mu[:, 2])
            # 创建伯努利分布并计算布尔变量的 log 概率
            dist_bool = torch.distributions.Bernoulli(probs=bool_prob)
            log_probs_bool = dist_bool.log_prob(actions[:, 2])

            # Combine log probabilities
            log_probs = log_probs_continuous + log_probs_bool

            # log_probs = F.mse_loss(logits, actions, reduction="none")
            pi_loss = -log_probs * advantages
            self.writer.add_scalar("Actor_loss",pi_loss.mean(),i)
            pi_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.actor_func.parameters(),GRAD_CLIP_NORM)
            self.opt2.step()

            # Output total rewards in episode (max 500)
            # print("Run episode{} with rewards {}".format(i, sum(rewards)))

        print("\nDone")
        self.env.close()

    def save(self, path):
        torch.save({
            'actor_func_state_dict': self.actor_func.state_dict(),
            'value_func_state_dict': self.value_func.state_dict(),
            'opt1_state_dict': self.opt1.state_dict(),
            'opt2_state_dict': self.opt2.state_dict(),
            'reward_records': self.reward_records,
            'best_score': self.best_score
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor_func.load_state_dict(checkpoint['actor_func_state_dict'])
        self.value_func.load_state_dict(checkpoint['value_func_state_dict'])
        self.opt1.load_state_dict(checkpoint['opt1_state_dict'])
        self.opt2.load_state_dict(checkpoint['opt2_state_dict'])
        self.reward_records = checkpoint['reward_records']
        self.best_score = checkpoint['best_score']

if __name__=="__main__":
    a2c= A2C()
    a2c.train()
