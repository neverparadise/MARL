from pettingzoo.magent import battlefield_v3
import numpy as np

def random_one_hot_policy(env, dones):
    # 기본적으로 dones에 존재하지 않는 에이전트들의 행동은 None으로 처리되어야 합니다.
    # dones에 에이전트가 존재하지 않는다는 것은 에이전트의 상태가 done이 되어 더 이상 action을 수행할 수 없음을 의미합니다.
    actions = {agent: None for agent in env.agents}

    # dones에 존재하는 에이전트들에게 랜덤한 에이전트를 할당합니다.
    for agent in dones.keys():
        number = np.random.randint(0,21)
        nb_classes = 21
        targets = np.array([number]).reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets]
        actions[agent] = one_hot_targets.squeeze(0)
        print(actions[agent])
    return actions

env = battlefield_v3.env(map_size=80, minimap_mode=False,
                     step_reward=-0.005, dead_penalty=-0.1,
                     attack_penalty=-0.1, attack_opponent_reward=0.2,
                     max_cycles=1000, extra_features=True)

env.reset()
env.render()

total_rewards = 0
episodes = 1
max_episodes = 0

max_cycles = 100
dones = {agent: False for agent in env.agents}

while (episodes < 5):
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        action = policy(observation, agent)
        env.step(action)
        total_rewards += sum(reward.values())
        env.render()