import argparse

import gym
from ma_gym.wrappers import Monitor

from Agent import Agent
from RandomAgent import RandomAgent

"""
Based on:
https://github.com/koulanurag/ma-gym/blob/master/examples/random_agent.py

This script executes the Pong simulator with two agents, both of which make
use of the ``Agent`` class. Nothing in this file needs to be changed, but
you can make changes for debugging purposes.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='PongDuel-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=550,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    # Set up environment
    env = gym.make(args.env)
    env = Monitor(env, directory='recordings/' + args.env, force=True)
    action_meanings = env.get_action_meanings()
    print(env.observation_space[0].shape)
    # Initialize agents
    my_agent = Agent(0, env.action_space[0].n, env.observation_space[0].shape)
    agents = [
        my_agent,
        RandomAgent(1)
    ]

    print(f'Action space: {env.action_space}')
    print(f'Observation (state) space: {env.observation_space}')

    wins = []
    losses = []
    # Run for a number of episodes
    for ep_i in range(args.episodes):
        are_done = [False for _ in range(env.n_agents)]
        ep_rewards = [0, 0]

        env.seed(ep_i)
        prev_observations = env.reset()
        env.render()

        while not all(are_done):
            # Observe:
            prev_observations = env.get_agent_obs()
            actions = []
            # For each agent, act:
            for (index, observation) in enumerate(prev_observations):
                action = agents[index].act(prev_observations)
                actions.append(action)
                # Use the command below to print the exact actions that the
                # agents are executing:
                # print([action_meanings[index][action] for action in actions])
            # Trigger the actual execution
            observations, rewards, are_done, infos = env.step(actions)
            # For each agent, update observations and rewards
            for agent in agents:
                agent.update_history(
                    prev_observations,
                    actions[agent.number],
                    rewards[agent.number],
                    observations
                )
            # Debug: print obs, rewards, info
            # print(obs_n, reward_n, info)
            for (index, reward) in enumerate(rewards):
                ep_rewards[index] += reward
            env.render()
        # Aggregate wins and losses
        bottom_line = ep_rewards[0]
        if bottom_line < 0:
            wins.append(0)
            losses.append(abs(bottom_line))
        else:
            wins.append(bottom_line)
            losses.append(0)
        print('Episode #{} Rewards: {}'.format(ep_i, ep_rewards))
        print(f'Wins - losses: {sum(wins) - sum(losses)}')
        print(f'Epsilon: {my_agent.epsilon}')
        print(f'Q table size: {len(my_agent.q)}')
        if len(wins) > 10:
            print(f'Last 10 games: {sum(wins[-10:]) - sum(losses[-10:])}')
    env.close()
