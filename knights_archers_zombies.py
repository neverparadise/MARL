from pettingzoo.butterfly import knights_archers_zombies_v7 as kaz7
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import numpy as np

env = kaz7.parallel_env(spawn_rate=20, num_archers=1, num_knights=2, killable_knights=True, killable_archers=True,
                                     pad_observation=True, line_death=True, max_cycles=900)

def make_actions(env, dones):
    # 기본적으로 dones에 존재하지 않는 에이전트들의 행동은 None으로 처리되어야 합니다.
    actions = {agent: None for agent in env.agents}

    # dones에 존재하는 에이전트들에게 랜덤한 에이전트를 할당합니다.
    for agent in dones.keys():
            actions[agent] = np.random.randint(0,6)
    return actions


env.reset()
max_cycles = 3000

# dones는 각 에이전트의 종료 정보를 담은 딕셔너리 입니다.
# 에이전트가 종료될 때마다 dones 에서 에이전트가 하나씩 빠지게 됩니다.
dones = {agent: False for agent in env.agents}
print(dones.keys())
print(dones.values())
for step in range(max_cycles):
    actions = make_actions(env, dones)
    observations, rewards, dones, infos = env.step(actions)
    print(observations)
    print(actions)
    print(dones)
    env.state()
    env.render()
    if not bool(dones):
        env.reset()
        dones = {agent: False for agent in env.agents}
