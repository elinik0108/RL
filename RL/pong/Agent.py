import numpy
import random
from collections import defaultdict


def reshape_obs(observation):
    """
    Reshapes and 'discretizes' an observation for Q-table read/write
    Make sure the state space is not too large!

    :param observation: The to-be-reshaped/discretized observation. Contains the position of the
    'players', as well as the position and movement.
    direction of the ball.
    :return: The reshaped/discretized observation
    """
    """
    Groups continuous positions into discrete bins so the Q-table 
    can actually find matches.
    """
    # Scaling by 10 and rounding to integers is a common way to 'bin' coordinates.
    # We convert to a tuple because dictionary keys must be hashable.
    return tuple(numpy.round(observation, 1))

class Agent:
    """
    Skeleton q-learner agent that the students have to implement
    """

    def __init__(
            self, id, actions_n, obs_space_shape,
            gamma=0.99, # Focuses on long-term rewards.
            epsilon=1.0, # Start with 100% exploration
            min_epsilon=0.01, # Always keep a 1% chance of acting randomly to handle unexpected situations.
            epsilon_decay=0.999, # A very slow decay to ensure the agent explores enough in the early episodes.
            alpha=0.1 # A moderate learning rate so the agent doesn't overreact to a single point.
    ):
        """
        Initiates the agent

        :param id: The agent's id in the game environment
        :param actions_n: The id of actions in the agent's action space
        :param obs_space_shape: The shape of the agents observation space
        :param gamma: Depreciation factor for expected future rewards
        :param epsilon: The initial/current exploration rate
        :param min_epsilon: The minimal/final exploration rate
        :param epsilon_decay: The rate of epsilon/exploration decay
        :param alpha: The learning rate
        """
        self.id = id
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.actions_n = actions_n
        self.obs_space_shape = obs_space_shape
        self.alpha = alpha
        self.q = defaultdict(lambda: numpy.zeros(self.actions_n))

    def determine_action_probabilities(self, observation):
        state = reshape_obs(observation)
        q_values = self.q[state]

        # Assign equal probability (epsilon / n) to all actions for exploration
        probs = numpy.ones(self.actions_n) * (self.epsilon / self.actions_n)
        
        # Find the best action from the Q-table
        best_action = numpy.argmax(q_values)
        
        # Add the remaining (1 - epsilon) probability to the best action for exploitation
        probs[best_action] += (1.0 - self.epsilon)        return probs
        """
        A function that takes the state as an input and returns the probabilities for each
        action in the form of a numpy array of length of the action space.
        :param observation: The agent's current observation
        :return: The probabilities for each action in the form of a numpy
        array of length of the action space.
        """
    def act(self, observation):
        # Get the epsilon-greedy probabilities
        probs = self.determine_action_probabilities(observation)
        
        # Choose an action based on the probability distribution
        action = numpy.random.choice(numpy.arange(self.actions_n), p=probs)
        
        # Decay epsilon: gradually move from exploring to exploiting
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)        return action
        """
        Determines and action, given the current observation.
        :param observation: the agent's current observation of the state of
        the world
        :return: the agent's action
        """

    def update_history(
            self, observation, action, reward, new_observation
    ):
        """
        Updates the agent's Q-table

        :param observation: The observation *before* the action
        :param action: The action that has been executed
        :param reward: The reward the action has yielded
        :param new_observation: The observation *after* the action
        :return:
        """
        # counterfactual next action, to later backpropagate reward to current action
        next_action = numpy.argmax(self.q[reshape_obs(new_observation)])
        td_target = reward + self.gamma * self.q[reshape_obs(new_observation)][next_action]
        td_delta = td_target - self.q[reshape_obs(observation)][action]
        self.q[reshape_obs(observation)][action] += self.alpha * td_delta

