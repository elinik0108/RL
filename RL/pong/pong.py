import argparse

import gym
from ma_gym.wrappers import Monitor
import matplotlib.pyplot as plt # plotting dependency

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
    parser = argparse.ArgumentParser(description='Pong simulator for ma-gym')
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
    win_loss_history = []
    plot_saved = False
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
                    actions[agent.id],
                    rewards[agent.id],
                    observations
                )
            # Debug: print obs, rewards, info
            # print(observations, rewards, infos)
            # Rewards are either 0 or [1, -1], or [-1, 1], try out by out-commenting the line below
            # Note that your agent is agent 0
            # if not (rewards[0] == 0 and rewards[1] == 0): print(rewards)
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
        
        current_differential = sum(wins) - sum(losses)
        win_loss_history.append(current_differential)
        
        print('Episode #{} Rewards: {}'.format(ep_i, ep_rewards))
        print(f'Wins - losses: {current_differential}')
        print(f'Epsilon: {my_agent.epsilon}')
        print(f'Q table size: {len(my_agent.q)}')
        if len(wins) > 10:
            print(f'Last 10 games: {sum(wins[-10:]) - sum(losses[-10:])}')


        ## plotting
        ax.clear()
        ax.plot(win_loss_history, linewidth=2, color='blue', label='Performance')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='Target: 1000')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Win-Loss diff', fontsize=12)
        ax.set_title('Agents win history', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.pause(0.01)

        if current_differential > 1000 and not plot_saved:
            filename = f'pong_performance_episode_{ep_i}_diff_{current_differential}.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            plot_saved = True

    # Save final plot at the end
    ax.clear()
    ax.plot(win_loss_history, linewidth=2, color='blue', label='Performance')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='Target: 1000')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Win-Loss difference', fontsize=12)
    ax.set_title('Agents win history', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig('pong_final_performance.png', dpi=150, bbox_inches='tight')
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep plot window open at the end
    
    env.close()