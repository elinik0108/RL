from random import random
from random import randrange
from random import gauss


def generate_reward(arm_index, expected_rewards_approx):
    """
    Adds some gaussian noise when generating rewards

    :param arm_index: Index of the current arm
    :param expected_rewards_approx: Reward approximation values for all arms
    :return: Reward + gaussian noise
    """
    return gauss(expected_rewards_approx[arm_index], 0) + random() / 2


def simulate(bandit, iterations):
    """
    Runs the provided bandit an `iterations` number of times, each time
    simulating a reward.

    :param bandit: The bandit that is to be simulated
    :param iterations: The number of iterations that the bandit should be run
    :return:
    """

    # Determine expected rewards
    expected_rewards_approx = [
        1 + (random() / 2) for _ in range(6)
    ]
    expected_rewards_approx[randrange(0,5)] = -12
    expected_rewards_approx[randrange(0,5)] = -20
    for (index, reward) in enumerate(expected_rewards_approx):
        expected_rewards_approx[index] = reward + (random() - 0.5) * reward * 0.75

    # Run bandit
    for _ in range(iterations):
        arm = bandit.run()
        reward = generate_reward(bandit.arms.index(arm), expected_rewards_approx)
        bandit.give_feedback(arm, reward)
    #print('Frequencies')
    #print(bandit.frequencies)
    return sum(bandit.sums)
