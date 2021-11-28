import random


class RandomAgent:
    """
    'Pong' agent that always takes a random action
    """

    def __init__(self, id):
        """
        Initiates the agent

        :param id: The agent's id in the game environment
        """
        self.id = id

    def act(self, observation):
        """
        Determines and action, given the current observation (and the state of
        the agent).
        :param observation: the agent's current observation of the state of
        the world
        :return: the agent's action
        """
        return random.randint(0,2)

    def update_history(
            self, observation, action, reward, new_observation
    ):
        """

        :param observation: The observation *before* the action
        :param action: The action that has been executed
        :param reward: The reward the action has yielded
        :param new_observation: The observation *after* the action
        """
        # Dummy function


