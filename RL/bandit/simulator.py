from random import random
from random import uniform
from random import randrange
from random import gauss
from numpy.random import normal, seed
seed(0)

def determine_rand_norm():
  """
  Determine random int, roughly normally distributed
  """
  number = round(normal(uniform(1,5), 2))
  if number not in range(0,6):
      return determine_rand_norm()
  return number

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
    expected_rewards_approx[0] = normal(uniform(-4,4), 2.5)
    expected_rewards_approx[determine_rand_norm()] = -12
    expected_rewards_approx[randrange(0,6)] = -5
    for (index, reward) in enumerate(expected_rewards_approx):
        expected_rewards_approx[index] = reward + (random() - 0.5) * reward * 0.75

    # Run bandit
    for _ in range(iterations):
        arm = bandit.run()
        current_reward = generate_reward(bandit.arms.index(arm), expected_rewards_approx)
        bandit.give_feedback(arm, current_reward)
    #print('Frequencies')
    #print(bandit.frequencies)
    return sum(bandit.sums)
