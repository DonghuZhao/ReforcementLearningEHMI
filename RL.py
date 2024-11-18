# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class Environment:
#     def __init__(self, num_states):
#         self.num_states = num_states
#         self.state = 0
#         self.goal = 10
#
#     def step(self, action):
#         # 执行动作并返回奖励和下一个状态
#         if action == 0 and self.state > 0:  # 向左移动
#             self.state -= 1
#         elif action == 1 and self.state < self.num_states - 1:  # 向右移动
#             self.state += 1
#
#         # 检查是否达到目标状态
#         if self.state == self.goal:
#             reward = 1
#             done = True
#         else:
#             reward = 0
#             done = False
#
#         # 如果向右移动，给予额外奖励
#         if action == 1:
#             reward += 0.1
#
#         # 返回奖励、下一个状态和是否终止的标志
#         return self.state, reward, done
#
#     def reset(self):
#         # 重置环境到初始状态
#         self.state = 0
#         return self.state
#
#
# class Agent:
#     def __init__(self, num_states, num_actions):
#         self.num_states = num_states
#         self.num_actions = num_actions
#         self.epsilon = 0.1  # ε-贪心策略中的探索率
#         self.Q = np.zeros((num_states, num_actions))
#
#     def select_action(self, state):
#         # ε-贪心策略
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.num_actions)
#         else:
#             return np.argmax(self.Q[state])
#
#     def update_Q(self, state, action, reward, next_state, alpha, gamma):
#         # 使用 Q-learning 更新 Q 值
#         self.Q[state][action] += alpha * (reward + gamma * np.max(self.Q[next_state]) - self.Q[state][action])
#
#
# # 设置参数
# num_states = 20
# num_actions = 2
# num_episodes = 1000
# alpha = 0.1  # 学习率
# gamma = 0.9  # 折扣因子
# print_freq = 20  # 输出训练进程的频率
#
# # 创建环境和智能体
# env = Environment(num_states)
# agent = Agent(num_states, num_actions)
#
# # 训练智能体
# episode_rewards = []
# episode_states = []
#
# for episode in range(num_episodes):
#     state = env.reset()
#     done = False
#     rewards = []
#     states = []
#     while not done:
#         action = agent.select_action(state)
#         next_state, reward, done = env.step(action)
#         agent.update_Q(state, action, reward, next_state, alpha, gamma)
#         state = next_state
#         rewards.append(reward)
#         states.append(state)
#     episode_rewards.append(sum(rewards))
#     episode_states.append(states)
#
#     print(f"Episode {episode}: Total Reward = {episode_rewards[-1]}")
#
# # 可视化训练过程
# plt.plot(episode_rewards)
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.title('Training Process')
# plt.show()
#
# # 可视化智能体的轨迹
# for episode in range(10):  # 绘制前10个轨迹
#     plt.plot(episode_states[episode], label=f'Episode {episode + 1}')
# plt.xlabel('Time Step')
# plt.ylabel('State')
# plt.title('Agent Trajectories')
# plt.legend()
# plt.show()


import torch
import torch.nn as nn

# 假设输入大小为 10，输出大小为 5
net = nn.Linear(10, 5)

# 假设当前状态的输入张量 x
x = torch.randn(1, 10)  # 生成一个大小为 [1, 10] 的随机张量

# 前向传播，得到动作值函数的估计值
actions_value = net(x)

# 找到预期回报最大的动作的索引
action_index = torch.max(actions_value, -1)[1].data.numpy()
print(torch.max(actions_value, -1))
print(torch.max(actions_value, -1)[1])

# 取出最大值
max_value = action_index.max()

print("动作值函数的估计值:", actions_value)
print("预期回报最大的动作索引:", action_index)
print("预期回报最大的动作索引的最大值:", max_value)

