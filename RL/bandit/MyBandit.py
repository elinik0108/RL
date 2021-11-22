# epsilon-greedy example implementation of a multi-armed bandit
import random

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class Bandit:
    """
    Generic epsilon-greedy bandit that you need to improve
    """

    def __init__(self, arms, epsilon=0.05):
        """
        Initiates the bandits

        :param arms: List of arms
        :param epsilon: Epsilon value for random exploration
        """
        self.arms = arms
        self.epsilon = epsilon
        self.frequencies = [0] * len(arms)
        self.sums = [0] * len(arms)
        self.expected_values = [0] * len(arms)

        self.rewards = [[] for i in range(len(arms))]
        self.nr = 10

    def run(self):
        """
        Asks the bandit to recommend the next arm

        :return: Returns the arm the bandit recommends pulling
        """
        if min(self.frequencies) == 0:
            return self.arms[self.frequencies.index(min(self.frequencies))]

        arms = [x for x in self.arms if self.expected_values[self.arms.index(x)] > 0]

        if random.random() < self.epsilon:
            return arms[random.randint(0, len(arms) - 1)]

        return self.arms[self.expected_values.index(max(self.expected_values))]

    def give_feedback(self, arm, reward):
        """
        Sets the bandit's reward for the most recent arm pull

        :param arm: The arm that was pulled to generate the reward
        :param reward: The reward that was generated
        """
        arm_index = self.arms.index(arm)

        self.rewards[arm_index].append(reward)

        frequency = self.frequencies[arm_index] + 1
        self.frequencies[arm_index] = frequency

        s = self.sums[arm_index] + reward
        self.sums[arm_index] = s

        if (frequency > self.nr):
            s = sum(self.rewards[arm_index][-self.nr:])
            frequency = self.nr

        expected_value = s / frequency
        self.expected_values[arm_index] = expected_value
