# epsilon-greedy example implementation of a multi-armed bandit
import random

class Bandit:
    """
    Generic epsilon-greedy bandit that you need to improve
    """
    def __init__(self, arms, epsilon=0.1, alpha= 0.1):
        """
        Initiates the bandits

        :param arms: List of arms
        :param epsilon: Epsilon value for random exploration
        :alpha : alpha value for random exploration
        """
        self.arms = arms
        self.epsilon = epsilon
        self.alpha = alpha

        self.frequencies = [0] * len(arms)
        self.sums = [0.0] * len(arms)
        self.expected_values = [0.0] * len(arms)

    def run(self):
        """
        Asks the bandit to recommend the next arm

        :return: Returns the arm the bandit recommends pulling
        """

        # checking that each arm is tried at least once
        for i, freq in enumerate(self.frequencies):
            if freq == 0:
                return self.arms[i]

        # explorations
        if random.random() < self.epsilon:
            return random.choice(self.arms)

        max_q = max(self.expected_values)

        ## create a list of all positions where value is equal to max
        best_index = []
        for i, q in enumerate(self.expected_values):
            if q == max_q:
                best_index.append(i)

        # return list of arms with randomly selected value from best_index
        return self.arms[random.choice(best_index)]

    def give_feedback(self, arm, reward):
        """
        Sets the bandit's reward for the most recent arm pull

        :param arm: The arm that was pulled to generate the reward
        :param reward: The reward that was generated
        """
        arm_index = self.arms.index(arm)

        # update frequency
        self.frequencies[arm_index] += 1

        self.sums[arm_index] += reward
        # unlike the standard epsilon greedy MAB that implicitly the reward destribution
        # of each arm does not change overtime. Here we update reward based on environmnet changes that
        # happens in "reward - self.expected_values[arm_index]"
        self.expected_values[arm_index] += self.alpha * (reward - self.expected_values[arm_index])
