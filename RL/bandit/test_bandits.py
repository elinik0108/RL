from ReferenceBandit import ReferenceBandit
from MyBandit import Bandit
import simulator as simulator
import random
import math


def simulate():
    """
    Simulates the two bandits and returns the simulation results

    :return: simulation results, list, for each entry:
             1 if bandit  beats reference bandit (* 1 + bonus); else 0

    """

    # configuration
    arms = [
        'Configuration a',
        'Configuration b',
        'Configuration c',
        'Configuration d',
        'Configuration e',
        'Configuration f'
    ]

    # instantiate bandits
    bandit = Bandit(arms)
    ref_bandit = ReferenceBandit(arms)
    results = []
    for index in range(0, 20):
        random.seed(index)
        iterations = int(math.floor(1000 * (random.random()) + 0.5))
        bandit_reward = simulator.simulate(bandit, iterations)
        ref_bandit_reward = simulator.simulate(ref_bandit, iterations)
        ref_plus_bonus = ref_bandit_reward * 1.35
        result = 0
        if (bandit_reward > ref_plus_bonus):
            result = 1
        results.append(result)
    return results


def test_performance():
    """
    Checks if the simulation is good enough to pass
    """
    assert sum(simulate()) > 15
