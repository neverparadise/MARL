
import random
import numpy as np
from smac.env.pettingzoo import StarCraft2PZEnv
from model.QMIX import *


def random_policy(env, obs,  dones, action_lens):
    # 기본적으로 dones에 존재하지 않는 에이전트들의 행동은 None으로 처리되어야 합니다.
    actions = {agent: None for agent in env.agents}
    print(obs)

    # dones에 존재하는 에이전트들에게 랜덤한 에이전트를 할당합니다.
    for agent in dones.keys():
        if dones[agent]:
            action = None
            actions[agent] = action
        elif isinstance(obs[agent], dict) and "action_mask" in obs[agent]:
            action = random.choice(np.flatnonzero(obs[agent]["action_mask"]))
            actions[agent] = action
        else:
            action = env.action_spaces[agent].sample()
            actions[agent] = action

    return actions

def print_messages(num_epi, steps, total_rewards):
    print("# {} episode is finished".format(num_epi))
    print("Currnet steps : {}".format(steps))
    print("Episode total rewards", total_rewards)

def main():
    """
    Runs an env object with random actions.
    """
    env = StarCraft2PZEnv.parallel_env(map_name="3m")
    max_steps = 100000
    steps = 0
    max_epi = 1000
    num_epi = 0
    action_lengths = None
    env.reset()
    print(env.agents)
    print(env.action_spaces)
    for agent in env.agents:
        print((env.action_spaces[agent].n))
        action_lengths = (env.action_spaces[agent].n)
        break
    print(env.observation_spaces)
    while steps < max_steps and num_epi < max_epi:
        t = 0
        total_rewards = 0
        obs = env.reset()
        dones = {agent: False for agent in env.agents}

        while True:
            steps += 1
            actions = random_policy(env, obs, dones, len(env.action_spaces))
            print(actions)
            print(dones)
            obs, rewards, dones, infos = env.step(actions)
            total_rewards += sum(rewards.values())
            env.render()

            # if done:
            #     action = None
            # elif isinstance(obs, dict) and "action_mask" in obs:
            #     action = random.choice(np.flatnonzero(obs["action_mask"]))
            # else:
            #     action = env.action_spaces[agent].sample()
            last_agent = list(dones)[0]

            if not False in dones.values():
                break

        print_messages(num_epi, steps, total_rewards)
        num_epi += 1



    env.close()

if __name__ == "__main__":
    main()