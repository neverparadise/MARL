from pettingzoo.magent import battlefield_v3

import numpy as np

def random_one_hot_policy(env, dones):
    # 기본적으로 dones에 존재하지 않는 에이전트들의 행동은 None으로 처리되어야 합니다.
    # dones에 에이전트가 존재하지 않는다는 것은 에이전트의 상태가 done이 되어 더 이상 action을 수행할 수 없음을 의미합니다.
    actions = {agent: None for agent in env.agents}

    # dones에 존재하는 에이전트들에게 랜덤한 에이전트를 할당합니다.
    for agent in dones.keys():
        number = np.random.randint(0,21)
        actions[agent] = number
    return actions



env2 = battlefield_v3.parallel_env(map_size=80, minimap_mode=False,
                     step_reward=-0.005, dead_penalty=-0.1,
                     attack_penalty=-0.1, attack_opponent_reward=0.2,
                     max_cycles=1000, extra_features=True)

env2.reset()
max_cycles = 1000
dones = {agent: False for agent in env2.agents}

total_rewards = 0
episodes = 1
max_episodes = 0

while (episodes < 5):
    for step in range(max_cycles):
        actions = random_one_hot_policy(env2, dones)
        #print(actions)

        observations, rewards, dones, infos = env2.step(actions)
        # print(rewards)
        total_rewards += sum(rewards.values())
        # print(dones)
        env2.render()
        a = env2.state()

        print("state")

        # 리스트나 딕셔너리는 비어 있으면 bool로 형변환은 했을 때 False가 됩니다.
        if not bool(dones):
            print("---------------------------------------------------------")
            print("{} episode is finised".format(episodes))
            print("Total rewards : {}".format(total_rewards))
            print("environment reset")
            print("---------------------------------------------------------")
            episodes += 1
            total_rewards = 0
            env2.reset()

            dones = {agent: False for agent in env2.agents}

env2.close()