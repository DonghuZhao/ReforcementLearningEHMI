import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import copy
import os
import sys
import json
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
MODEL_UPDATE_EPOCHS = 20  # 10
TIME = time.strftime("%Y%m%d%H%M%S",time.localtime())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.μ = nn.Linear(256, action_size)
        self.σ = nn.Linear(256, action_size)
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_p)

        # to cuda
        self.to(device)

    def forward(self, x):
        temp = x
        x = self.net(x)
        μ = torch.tanh(self.μ(x)) * 1
        σ = F.softplus(self.σ(x)) + 1e-2           # 1e-7
        if torch.isnan(μ).any() or torch.isnan(σ).any():
            print(f'nan generated by {temp}')
            print('x:', x)
        return μ, σ

    def check_nan_parameters_in_net(self):
        for name, parameter in self.net.named_parameters():
            if torch.isnan(parameter).any() or torch.isinf(parameter).any():
                print(f"NaN found in {name} with value {parameter}")
                return True
        # print("No NaN in net parameters")
        return False

class DiscretePolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super(DiscretePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.action_logits = nn.Linear(256, 3)  # 输出动作的 logits，动作空间大小为 3
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_p)

        # to cuda
        self.to(device)

    def forward(self, x):
        x = self.net(x)
        action_logits = self.action_logits(x)
        return action_logits

    def sample_action(self, x):
        action_logits = self.forward(x)
        action_probs = F.softmax(action_logits, dim=-1)  # 计算动作概率
        action_dist = torch.distributions.Categorical(action_probs)  # 创建动作分布
        action = action_dist.sample()  # 采样动作
        return action.item()  # 返回动作的标量值

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
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_v)

        self.to(device)
 
    def forward(self, x):
        x = self.net(x)
        return x
    
class Agent(object):
    """
    Agent:
    env:            用于模型训练的环境;
    agent_path:     当env为空时，用于读取agent的参数;
    state_size:     状态空间的大小;
    action_size:    动作空间的大小;
    v:              价值网络;
    p:              策略网络;
    old_v:          旧的价值网络;
    old_p:          旧的策略网络;
    step:           模型训练时的步数;
    max_average_rewards: 单个场景的最大奖励;
    last_update_step: 上一次更新网络时的步数;
    """
    def __init__(self, env=None, agent_path=r'.\model\agent.json'):
        # 加载Agent参数
        self.agent_path = agent_path
        if env:
            self.env = env
            self.isHaveEHMI = self.env.get_wrapper_attr('config')["action"]["EHMI"]
            # self.state_size = int(self.env.observation_space.shape[0] * self.env.observation_space.shape[1])
            # self.state_size = self.state_size + 4 if self.isHaveEHMI else self.state_size
            self.state_size = 10
            self.state_size = self.state_size + 3 if self.isHaveEHMI else self.state_size
            # self.state_size = 14 # 雷达信息v2
            # self.state_size += 3 # 雷达信息v1
            self.action_size = self.env.action_space.shape[0]
        else:
            self.loadAgentParas(self.agent_path)

        self.v = Value(self.state_size + 1).to(device)
        self.p = Policy(self.state_size + 1, self.action_size).to(device)
        self.old_p = Policy(self.state_size + 1, self.action_size).to(device)      #旧策略网络
        self.old_v = Value(self.state_size + 1).to(device)         #旧价值网络    用于计算上次更新与下次更新的差别

        self.v_level0 = Value(self.state_size).to(device)
        self.p_level0 = DiscretePolicy(self.state_size, 1).to(device)
        self.old_p_level0 = DiscretePolicy(self.state_size, 1).to(device)
        self.old_v_level0 = Value(self.state_size).to(device)

        self.data_level0 = []
        self.data = []               #用于存储经验
        self.step = 0

        self.max_average_rewards = -1e6
        self.average_rewards = 0
        self.sum_rewards = 0

        self.initial_epsilon = 0.2
        self.min_epsilon = 0.01

        self.writer = SummaryWriter(comment=TIME)

        self.last_update_step = 0
    
    def choose_action(self, s):
        with torch.no_grad():
            s = s.to(device)
            μ, σ = self.old_p(s)
            actions = []
            # self.step = 10000
            epsilon = self.initial_epsilon * (self.min_epsilon / self.initial_epsilon) ** (self.step / MAX_EPOCHS)
            epsilon = max(epsilon, 0.01) if self.step < 2000 else 0
            # print(epsilon)
            for i in range(self.action_size):
                distribution = torch.distributions.Normal(μ[i], σ[i])
                action = distribution.sample()
                # # 以一定的概率添加噪声
                # if random.random() < epsilon:
                #     noise = torch.randn(1) * NOISE_SCALE
                #     action += noise[0]
                #     # 确保动作在合法范围内
                #     action = torch.clamp(action, -0.5, 0.5)
                #     # 将 action 转换回零维张量
                #     action = action.squeeze()
                # print(action)
                actions.append(action.item())
        return actions

    def choose_action_level0(self, s):
        with torch.no_grad():
            s = s.to(device)
            EHMI = self.p_level0(s)
        return EHMI.argmax().item()

    def translateEHMI(self, value):
        if value > 0.5:
            return 'R'
        return 'Y'

    def push_data(self, transitions_level0, transitions):
        self.data_level0.append(transitions_level0)
        self.data.append(transitions)
 
    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.data:
            s, a, r, s_, done = item
            l_s.append(torch.tensor(np.array([s]), dtype=torch.float))
            l_a.append(torch.tensor(np.array([a]), dtype=torch.float))
            l_r.append(torch.tensor(np.array([[r]]), dtype=torch.float))
            l_s_.append(torch.tensor(np.array([s_]), dtype=torch.float))
            l_done.append(torch.tensor(np.array([[done]]), dtype=torch.float))
        s = torch.cat(l_s, dim=0).to(device)
        a = torch.cat(l_a, dim=0).to(device)
        r = torch.cat(l_r, dim=0).to(device)
        s_ = torch.cat(l_s_, dim=0).to(device)
        done = torch.cat(l_done, dim=0).to(device)
        self.data = []
        return s, a, r, s_, done

    def sample_level0(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.data_level0:
            s, a, r, s_, done = item
            l_s.append(torch.tensor(np.array([s]), dtype=torch.float))
            l_a.append(torch.tensor(np.array([a]), dtype=torch.float))
            l_r.append(torch.tensor(np.array([[r]]), dtype=torch.float))
            l_s_.append(torch.tensor(np.array([s_]), dtype=torch.float))
            l_done.append(torch.tensor(np.array([[done]]), dtype=torch.float))
        s = torch.cat(l_s, dim=0).to(device)
        a = torch.cat(l_a, dim=0).to(device)
        r = torch.cat(l_r, dim=0).to(device)
        s_ = torch.cat(l_s_, dim=0).to(device)
        done = torch.cat(l_done, dim=0).to(device)
        self.data_level0 = []
        return s, a, r, s_, done

    def update(self):
        self.step += 1
        # print("step:", self.step)
        s, a, r, s_, done = self.sample()
        for _ in range(K_epoch):
            with torch.no_grad():
                
                '''用于计算价值网络loss'''
                td_target = r + GAMMA * self.old_v(s_) * (1 - done)
                
                
                '''用于计算策略网络loss'''
                μ, σ = self.old_p(s)
                log_prob_old = 0
                for i in range(self.action_size):
                    μ_i = μ[:, i].unsqueeze(1)
                    σ_i = σ[:, i].unsqueeze(1)
                    old_dist_i = torch.distributions.Normal(μ_i, σ_i)
                    a_i = a[:, i].unsqueeze(1)
                    log_prob_old += old_dist_i.log_prob(a_i)

                td_error = r + GAMMA * self.v(s_) * (1 - done) - self.v(s)
                A = []
                adv = 0.0
                if device.type == 'cuda':
                    for td in td_error.flip(dims=[0]):  # 使用flip代替[::-1]来反转Tensor
                        adv = adv * GAMMA * LAMBDA + td
                        A.append(adv)
                else:
                    td_error = td_error.detach().numpy()
                    for td in td_error[::-1]:
                        adv = adv * GAMMA * LAMBDA + td[0]
                        A.append(adv)
                A.reverse()
                A = torch.tensor(A, dtype=torch.float, device=device).reshape(-1, 1)

            μ, σ = self.p(s)
            log_prob_new = 0
            for i in range(self.action_size):
                μ_i = μ[:, i].unsqueeze(1)
                σ_i = σ[:, i].unsqueeze(1)
                new_dist_i = torch.distributions.Normal(μ_i, σ_i)
                a_i = a[:, i].unsqueeze(1)
                log_prob_new += new_dist_i.log_prob(a_i)


            ratio = torch.exp(log_prob_new - log_prob_old)

            L1 = ratio * A
            L2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A
            loss_p = -torch.min(L1, L2).mean()
            self.p.optim.zero_grad()
            loss_p.backward()
            self.check_gradients()
            self.p.check_nan_parameters_in_net()
            self.p.optim.step()
 
            loss_v = F.huber_loss(td_target.detach(), self.v(s))
            self.v.optim.zero_grad()
            loss_v.backward()
            self.v.optim.step()

            self.writer.add_scalar("Actor_loss", loss_p.mean(), self.step)
            self.writer.add_scalar("Critic_loss", loss_v.mean(), self.step)

        self.old_p.load_state_dict(self.p.state_dict())
        self.old_v.load_state_dict(self.v.state_dict())

    def update_level0(self):
        s, a, r, s_, done = self.sample_level0()
        for _ in range(K_epoch):
            with torch.no_grad():

                '''用于计算价值网络loss'''
                td_target = r + GAMMA * self.old_v(s_) * (1 - done)

                '''用于计算策略网络loss'''
                action_logits = self.old_p_level0(s)
                old_dist = Categorical(logits=action_logits)
                log_prob_old = old_dist.log_prob(a)

                td_error = r + GAMMA * self.v_level0(s_) * (1 - done) - self.v_level0(s)
                A = []
                adv = 0.0
                if device.type == 'cuda':
                    for td in td_error.flip(dims=[0]):  # 使用flip代替[::-1]来反转Tensor
                        adv = adv * GAMMA * LAMBDA + td
                        A.append(adv)
                else:
                    td_error = td_error.detach().numpy()
                    for td in td_error[::-1]:
                        adv = adv * GAMMA * LAMBDA + td[0]
                        A.append(adv)
                A.reverse()
                A = torch.tensor(A, dtype=torch.float, device=device).reshape(-1, 1)

            action_logits = self.p_level0(s)
            new_dist = Categorical(logits=action_logits)
            log_prob_new = new_dist.log_prob(a)

            ratio = torch.exp(log_prob_new - log_prob_old)

            L1 = ratio * A
            L2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A
            loss_p = -torch.min(L1, L2).mean()
            self.p_level0.optim.zero_grad()
            loss_p.backward()
            self.check_gradients()
            self.p_level0.check_nan_parameters_in_net()
            self.p_level0.optim.step()

            loss_v = F.huber_loss(td_target.detach(), self.v_level0(s))
            self.v_level0.optim.zero_grad()
            loss_v.backward()
            self.v_level0.optim.step()

            self.writer.add_scalar("Actor_loss", loss_p.mean(), self.step)
            self.writer.add_scalar("Critic_loss", loss_v.mean(), self.step)

        self.old_p_level0.load_state_dict(self.p_level0.state_dict())
        self.old_v_level0.load_state_dict(self.v_level0.state_dict())

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
        # self.loadAgentParas(self.agent_path)
        # self.loadNetParas()
        # print('state_size:', self.state_size)
        # print('action_size:', self.action_size)
        for count in range(MAX_EPOCHS):

            s = self.env.reset()[0]
            # print(s)
            # s = self.env.reset()[0][:9]
            done = False
            rewards = 0

            while not done:
                # print(s)
                # s = np.array(s)
                EHMI = self.choose_action_level0(torch.tensor(s, dtype=torch.float))
                EHMI_dict = {0: None, 1: 'R', 2: 'Y'}
                self.env.unwrapped.update_EHMI(EHMI_dict[EHMI])
                sub_s = np.append(s, EHMI)
                a = self.choose_action(torch.tensor(sub_s, dtype=torch.float))
                s_, r, done, truncated, _ = self.env.step(a)
                sub_s_ = np.append(s_, EHMI)

                # self.env.render()

                rewards += r

                self.push_data((s, EHMI, r, s_, done), (sub_s, a, r, sub_s_, done))
                s = s_
                if truncated:
                    break

            self.sum_rewards += rewards
            self.update()

            if count > 0 and count % MODEL_UPDATE_EPOCHS == 0:       # 10

                self.average_rewards = self.sum_rewards / MODEL_UPDATE_EPOCHS
                self.sum_rewards = 0
                print(count - MODEL_UPDATE_EPOCHS + 1, '-', count, 'average_rewards:', self.average_rewards, 'max_average_rewards:',
                      self.max_average_rewards, 'last_update_epoch:', self.last_update_step)
                self.writer.add_scalar("Avg_Rewards_10", self.average_rewards, count)

                if self.max_average_rewards < self.average_rewards:
                    self.max_average_rewards = self.average_rewards
                    self.saveNetParas()
                    self.saveAgentParas()
                    self.last_update_step = self.step

    def saveNetParas(self):
        """save parameters of deep neural networks"""
        torch.save(self.p.state_dict(), r'.\model\p.pth')
        torch.save(self.v.state_dict(), r'.\model\v.pth')
        print('...save model...')
 
    def loadNetParas(self):
        """load parameters of deep neural networks"""
        try:
            self.p.load_state_dict(torch.load(r'.\model\p.pth'))
            self.v.load_state_dict(torch.load(r'.\model\v.pth'))
            print('...load...')
        except:
            pass

    def _agent_todict(self):
        """save agent parameters in dict"""
        return {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'EHMI': self.isHaveEHMI,
            'gamma': GAMMA,
            'lr_p': LR_p,
            'lr_v': LR_v,
            'k_epoch': K_epoch,
            'exploration_rate': EXPLORATION_RATE,
            'noise_scale': NOISE_SCALE,
            'max_epochs': MAX_EPOCHS,
            'max_average_reward': self.max_average_rewards,
            'features_range': self.env.unwrapped.config['observation']['features_range'],
            'step': self.step,
        }

    def saveAgentParas(self):
        """save parameters of Agent setting"""
        params = self._agent_todict()
        with open(r'.\model\agent.json', 'w') as f:
            json.dump(params, f)

    def loadAgentParas(self, agent_path):
        """load parameters of Agent setting"""
        with open(agent_path, 'r') as f:
            params = json.load(f)
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.isHaveEHMI = params['EHMI']
        self.max_average_rewards = params['max_average_reward']
        self.features_range = params['features_range']
        self.step = params['step']