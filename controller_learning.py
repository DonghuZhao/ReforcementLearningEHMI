import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA=0.99
LR=0.001
MAX_EPS = 200000
GRAD_CLIP_NORM = 5
TIME = time.strftime("%Y%m%d%H%M%S",time.localtime())

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
        actions_mu = F.tanh(self.output(outs))
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
        self.action_size = self.env.action_space.shape[0]
        self.actor_func = ActorNet(self.state_size,self.action_size).to(device)
        self.value_func = ValueNet(self.state_size).to(device)
        self.gamma = GAMMA
        self.reward_records = []
        self.opt1 = torch.optim.AdamW(self.value_func.parameters(), lr=LR)
        self.opt2 = torch.optim.AdamW(self.actor_func.parameters(), lr=LR)
        self.best_score = None
        self.writer = SummaryWriter(comment='-'+"Obstacle_avoidance" +TIME)
        self.exploration = 0.1
        # 创建一个全局变量来控制渲染状态
        self.rendering = False
        self.running = False

    # pick up action with above distribution policy_pi
    def pick_sample(self, s):
        with torch.no_grad():
            # observation降维
            s = s.reshape(-1,self.state_size)
            s = s.squeeze(0)
            # 增加维度
            s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
            # Get logits from state
            logits = self.actor_func(s_batch)
            # logits 的形状是 (1, 1, 2)
            means = logits.squeeze(dim=0)
            if np.random.random() < self.exploration:
                means = torch.tensor([np.random.random() * 2 - 1, np.random.random() * 2 - 1], dtype=torch.float).to(device)
            # 设定标准差，可以是固定值，也可以是通过网络输出
            std = torch.tensor([0.1, 0.1], dtype=torch.float).to(device)
            # 从高斯分布中采样连续变量
            a = torch.normal(means, std)
            # print("a:", a)
            # Return
            return a.tolist()

    def train(self, path):
        # self.load(path)
        for i in trange(MAX_EPS):
            #
            # Run episode till done
            #
            if i != 0 and i % 100 == 0:
                self.save(path)
            if i < MAX_EPS/2:
                self.exploration = i / MAX_EPS
            done = False
            states = []
            actions = []
            rewards = []
            s= self.env.reset()[0]
            while not done:
                states.append(s.reshape(1,self.state_size).tolist())
                a = self.pick_sample(s)
                # 常规步骤
                s, r, done, truncated, info = self.env.step(a)
                # print("state:", s)
                # 是否显示Highway的可视化界面
                # self.env.render()
                # actions.append(np.expand_dims(a, axis=0))
                actions.append(a)
                rewards.append(r)

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
            mu = actions_mu
            std = torch.tensor([[0.1, 0.1] for _ in range(len(mu))])
            # Create normal distribution and compute log probabilities
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(-1)  # Summing log probabilities across action dimensions

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
