import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# from pathplanning_env import pathplanning
import time
import copy
import matplotlib.pyplot as plt
import os
import sys
import random
os.chdir(sys.path[0])

LR_v = 1e-5
LR_p = 1e-5
K_epoch = 8
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
MAX_EPOCHS = 20000
EXPLORATION_RATE = 0.1
NOISE_SCALE = 0.1
TIME = time.strftime("%Y%m%d%H%M%S",time.localtime())

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.out = nn.Linear(256, action_size)  # 输出三个动作的概率
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_p)

    def forward(self, x):
        x = self.net(x)
        # print(x)
        # print(self.out(x))
        prob = F.softmax(self.out(x), dim=0)  # 使用softmax获取概率分布
        # print(prob)
        return prob

    def check_nan_parameters_in_net(self):
        for name, parameter in self.net.named_parameters():
            if torch.isnan(parameter).any() or torch.isinf(parameter).any():
                print(f"NaN found in {name} with value {parameter}")
                return True
        # print("No NaN in net parameters")
        return False


class Value(nn.Module):
    def __init__(self, state_size):
        super(Value, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_v)
 
    def forward(self, x):
        x = self.net(x)
        return x
    
class Agent(object):
    def __init__(self, env):
        self.env = env
        self.isHaveEHMI = self.env.get_wrapper_attr('config')["action"]["EHMI"]
        self.state_size = int(self.env.observation_space.shape[0] * self.env.observation_space.shape[1])
        self.state_size = self.state_size + 4 if self.isHaveEHMI else self.state_size
        # self.state_size = 9
        self.action_size = 3
        self.v = Value(self.state_size)
        self.p = Policy(self.state_size, self.action_size)
        self.old_p = Policy(self.state_size, self.action_size)        #旧策略网络
        self.old_v = Value(self.state_size)         #旧价值网络    用于计算上次更新与下次更新的差别

        self.data = []               #用于存储经验
        self.step = 0

        self.max_average_rewards = -1e6
        self.average_rewards = 0
        self.sum_rewards = 0

        self.initial_epsilon = 1
        self.min_epsilon = 0.01

        self.writer = SummaryWriter(comment=TIME)

        self.last_update_step = 0

    def choose_action(self, s):
        with torch.no_grad():
            prob = self.old_p(s)
            action = torch.argmax(prob)  # 选择概率最高的动作
            # action = action.item()
        return [action]  # 返回一个包含单个动作的列表

    def translateEHMI(self, value):
        if value > 0.5:
            return 'R'
        return 'Y'

    def push_data(self, transitions):
        self.data.append(transitions)
 
    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.data:
            s, a, r, s_, done = item
            l_s.append(torch.tensor([s], dtype=torch.float))
            l_a.append(torch.tensor([a], dtype=torch.float))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_s_.append(torch.tensor([s_], dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        self.data = []
        return s, a, r, s_, done

    def update(self):
        self.step += 1
        s, a, r, s_, done = self.sample()
        for _ in range(K_epoch):
            with torch.no_grad():
                # 用于计算价值网络loss
                td_target = r + GAMMA * self.old_v(s_) * (1 - done)

                # 用于计算策略网络loss
                prob = self.old_p(s)  # 获取旧策略下的动作概率分布
                advantage = r + GAMMA * self.v(s_) * (1 - done) - self.v(s)  # 计算优势函数
                advantage = advantage.detach()  # 阻止优势函数的梯度回传

            # 计算新策略下的动作概率分布
            prob_new = self.p(s)
            prob_old = self.old_p(s)
            # print('prob_new:\n', prob_new, 'a:\n', a, 'advantage:\n', advantage)
            # 计算策略网络的损失，使用交叉熵损失乘以优势函数
            # policy_loss = F.cross_entropy(prob_new, a) * advantage
            policy_loss = F.cross_entropy(prob_new, prob_old) * advantage

            # 计算价值网络损失
            value_loss = F.huber_loss(td_target, self.v(s))

            # 优化策略网络
            self.p.optim.zero_grad()
            policy_loss.mean().backward()  # 取平均后回传梯度
            self.p.optim.step()

            # 优化价值网络
            self.v.optim.zero_grad()
            value_loss.mean().backward()  # 取平均后回传梯度
            self.v.optim.step()

            # 记录损失
            self.writer.add_scalar("Actor_loss", policy_loss.mean().item(), self.step)
            self.writer.add_scalar("Critic_loss", value_loss.mean().item(), self.step)

        # 更新旧策略和价值网络
        self.old_p.load_state_dict(self.p.state_dict())
        self.old_v.load_state_dict(self.v.state_dict())

    def check_gradients(self):
        for name, param in self.p.named_parameters():
            if param.grad is not None:
                max_grad = param.grad.data.abs().max()
                if max_grad > 100:  # 设置一个阈值，例如1e6
                    # print(f"Gradient explosion detected in {name} with max grad: {max_grad}")
                    torch.nn.utils.clip_grad_norm_(self.p.parameters(), max_norm=100.0)  # 梯度裁剪

        # 检查参数值是否包含NaN或inf，并将其设置为0
        for name, param in self.p.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"NaN or inf detected in {name}, resetting to 0")
                param.data.fill_(0)
    def train(self):
        # self.load()
        # print('state_size:', self.state_size)
        # print('action_size:', self.action_size)
        # print(self.env.get_wrapper_attr('config')["duration"])
        for count in range(MAX_EPOCHS):

            s = self.env.reset()[0]
            # print(s)
            # s = self.env.reset()[0][:9]
            done = False
            rewards = 0
            # plot = np.zeros((2, 1000))

            while not done:
                # print(s)
                # s = np.array(s)

                # plot[0, i] = s[0]
                # plot[1, i] = s[1]

                a = self.choose_action(torch.tensor(s, dtype=torch.float))
                s_, r, done, truncated, _ = self.env.step(a[0])  # 传递单个动作
                a_ = copy.deepcopy(a)
                if self.isHaveEHMI:
                    self.env.unwrapped.EHMI = self.translateEHMI(a_[2])
                    a_ = a_[:2]
                # s_, r, done, truncated, _ = self.env.step(a_)
                # s_ = s_[:9]
                self.env.render()

                rewards += r

                self.push_data((s, a, r, s_, done))
                s = s_
                if truncated:
                    break

            self.sum_rewards += rewards
            self.update()
            # try:
            #     self.update()
            # except Exception as e:
            #     print(f'update error:{e}')
            #     continue
            if count > 0 and count % 10 == 0:

                self.average_rewards = self.sum_rewards / 10
                self.sum_rewards = 0
                print(count - 9, '-', count, 'average_rewards:', self.average_rewards, 'max_average_rewards:',
                      self.max_average_rewards, 'last_update_epoch:', self.last_update_step)
                self.writer.add_scalar("Avg_Rewards_10", self.average_rewards, count)

                if self.max_average_rewards < self.average_rewards:
                    self.max_average_rewards = self.average_rewards
                    self.save()
                    self.last_update_step = self.step

    def save(self):
        torch.save(self.p.state_dict(), r'.\model\p.pth')
        torch.save(self.v.state_dict(), r'.\model\v.pth')
        print('...save model...')
 
    def load(self):
        try:
            self.p.load_state_dict(torch.load(r'.\model\p.pth'))
            self.v.load_state_dict(torch.load(r'.\model\v.pth'))
            print('...load...')
        except:
            pass