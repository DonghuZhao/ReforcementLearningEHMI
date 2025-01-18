import gymnasium as gym
from matplotlib import pyplot as plt
import os

import PPO
import PPO_discrete

def env_test(env):
    # 绘制场景开始时的画面
    plt.imshow(env.render())
    plt.show()
    print('-----start----------')
    # 场景运行
    done = False
    truncated = False
    while not done and not truncated:
        action = env.unwrapped.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        print(obs)
        print(done, truncated)
        env.render()
    # 绘制场景结束时的画面
    plt.imshow(env.render())
    plt.show()
    print('-----end----------')


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    ENV_TEST = False

    # 环境参数设置
    env = gym.make('intersection-v1', render_mode='rgb_array')
    env.get_wrapper_attr('config')["vehicles_count"] = 2
    env.get_wrapper_attr('config')["duration"] = 20 # 30
    env.get_wrapper_attr('config')["observation"]["vehicles_count"] = 2
    # configuration for environment test
    if ENV_TEST:
        env.get_wrapper_attr('config')["action"]["type"] = "DiscreteMetaAction"
        env.get_wrapper_attr('config')["action"]["lateral"] = False
        env.get_wrapper_attr('config')["action"]["EHMI"] = False

        # 环境参数更新
        env.reset()
        # 环境测试
        env_test(env)

    else:
        # 模型训练
        path = 'model/DRL_Highway.pkl'
        ppo = PPO.Agent(env)
        ppo.train()
        # ppo.saveAgentParas()