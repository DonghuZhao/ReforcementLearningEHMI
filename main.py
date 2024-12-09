from PPO import Agent

# 1. load model
agent = Agent(agent_path=r'.\model\agent.json')
agent.loadNetParas()

# "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "lat_off", "ang_off"],
# "features": ["x", "y", "vx", "vy","lat_off", "ang_off"],        # state_v3
# "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "lat_off", "ang_off", "EHMI", "Safety"],

# "features_range": {
#     "x": [-20, 60],
#     "y": [-50, 30],
#     "vx": [-10, 10],
#     "vy": [-10, 10],
#     "cos_h": [-1, 1],
#     "sin_h": [-1, 1],
#     "lat_off": [-2, 3],
#     "ang_off": [-np.pi / 4, np.pi / 4],
#     "Safety": [0, 100],

while True:
    s = input()
    a = agent.choose_action(s)
