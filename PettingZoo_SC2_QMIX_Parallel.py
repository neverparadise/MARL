
import random
import numpy as np
from smac.env.pettingzoo import StarCraft2PZEnv
from model.QMIX import *


#class QLearner:
#    def __init__(self):


def random_policy(env, obs,  dones, action_lens):
    # 기본적으로 dones에 존재하지 않는 에이전트들의 행동은 None으로 처리되어야 합니다.
    actions = {agent: None for agent in env.agents}

    # dones에 존재하는 에이전트들에게 랜덤한 에이전트를 할당합니다.
    for agent in dones.keys():
        if dones[agent]:
            action = None
            actions[agent] = action
        elif isinstance(obs[agent], dict) and "action_mask" in obs[agent]:
            action = random.choice(np.flatnonzero(obs[agent]["action_mask"]))
            actions[agent] = action
        else:
            print("what is this code?")
            action = env.action_spaces[agent].sample()
            actions[agent] = action

    return actions

def print_messages(num_epi, steps, total_rewards):
    print("# {} episode is finished".format(num_epi))
    print("Currnet steps : {}".format(steps))
    print("Episode total rewards", total_rewards)

def make_tau(obs, pre_action):
    pass


def main():
    # Network Initialization




    """
    Runs an env object with random actions.
    """
    env = StarCraft2PZEnv.parallel_env(map_name="3m")

    max_steps = 100000
    steps = 0
    max_epi = 1000
    num_epi = 0
    start_epsilon = 0.95

    action_lengths = None
    env.reset()
    print(env.agents)
    print(env.action_spaces)
    print(env.observation_spaces)

    return

    Q_Agents = {agent: RNNAgent(input_shape=, hidden_dim=32, num_actions=) for agent in env.agents}

    for agent in env.agents:
        print((env.action_spaces[agent].n))
        action_lengths = (env.action_spaces[agent].n)
        break

    # Epsilon Scheduling
    startEpsilon = 1.0
    endEpsilon = 0.05
    epsilon = startEpsilon
    stepDrop = (startEpsilon - endEpsilon) / max_epi

    while steps < max_steps and num_epi < max_epi:
        total_rewards = 0
        dones = {agent: False for agent in env.agents}

        # Initialize tau_0
        obs = env.reset()
        print(obs)
        pre_actions = {agent: None for agent in env.agents}
        tau = (obs, pre_actions)
        print
        while True:


            # Schedule epsilon
            if (epsilon > endEpsilon):
                epsilon -= stepDrop

            # Select action for each agent
            actions = Policy()

            # Combine pre_tau and obs, pre_actions

            actions = random_policy(env, obs, dones, len(env.action_spaces))
            # print(obs)
            # print(actions)
            # print(dones)

            # Get reward, next_state
            tau, rewards, dones, infos = env.step(actions)
            total_rewards += sum(rewards.values())
            steps += 1

            # Replay Buffer Updates
            # --- implement required


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