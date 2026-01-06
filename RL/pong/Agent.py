import numpy
import random
from collections import defaultdict


def reshape_obs(observation):
    """
    Reshapes and 'discretizes' an observation for Q-table read/write
    Make sure the state space is not too large!

    :param observation: The to-be-reshaped/discretized observation. Contains the position of the
    'players', as well as the position and movement direction of the ball.
    :return: The reshaped/discretized observation
    """
    obs = numpy.asarray(observation).flatten()
    
    # Extract relevant information from observation
    # Assuming observation structure: [player1_y, player2_y, ball_x, ball_y, ball_vx, ball_vy, ...]
    # You may need to adjust indices based on your actual observation structure
    
    # Agents position on y axel. Agent is moving always on y axel so we don't need x.
    paddle_y = obs[0]
    
    # Ball position
    ball_x = obs[2]
    ball_y = obs[3]
    
    # Ball vel on x axel
    ball_vx = obs[4]

    # Ball vel on y axel
    ball_vy = obs[5]
    
    # Discretize positions 
    discretization_factor = 10
    
    disc_paddle_y = int((paddle_y* 100) / discretization_factor)
    disc_ball_x = int((ball_x *100) / discretization_factor)
    disc_ball_y = int((ball_y *100)/ discretization_factor)
    
    ball_dir_x = numpy.sign(ball_vx)
    ball_dir_y = numpy.sign(ball_vy)
    
    # Create state tuple
    state = (disc_paddle_y, disc_ball_x, disc_ball_y, ball_dir_x, ball_dir_y)
    
    return state
    # Scaling by 10 and rounding to integers is a common way to 'bin' coordinates.
    # We convert to a tuple because dictionary keys must be hashable.
    # return tuple(numpy.round(observation, 1))

    #return f'{numpy.asarray(observation).reshape(-1, 10)}'

class Agent:
    """
    Skeleton q-learner agent that the students have to implement
    """

    def __init__(
            
            ## based on script these values had best performance
            self, id, actions_n, obs_space_shape,
            gamma=0.9,
            epsilon=1.0,
            min_epsilon=0.01,
            epsilon_decay=0.995,
            alpha=0.05
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

        self.step_counter = 0

    def determine_action_probabilities(self, observation):
        """
        A function that takes the state as an input and returns the probabilities for each
        action in the form of a numpy array of length of the action space.
        
        Uses epsilon-greedy strategy:
        - With probability epsilon: uniform random action (exploration)
        - With probability 1-epsilon: best action according to Q-table (exploitation)
        
        :param observation: The agent's current observation
        :return: The probabilities for each action in the form of a numpy
        array of length of the action space.
        """
        # Get the discretized state
        state = reshape_obs(observation)
        
        # Get Q-values for this state
        q_values = self.q[state]
        
        # Find the highest q-value
        best_action = numpy.argmax(q_values)
        
        # Initialize probabilities array
        action_probabilities = numpy.ones(self.actions_n) * (self.epsilon / self.actions_n)
        
        ## the best action should have the highest probability
        action_probabilities[best_action] += (1.0 - self.epsilon)
        
        return action_probabilities
        """
        A function that takes the state as an input and returns the probabilities for each
        action in the form of a numpy array of length of the action space.
        :param observation: The agent's current observation
        :return: The probabilities for each action in the form of a numpy
        array of length of the action space.
        """
    def act(self, observation):
        """
        Determines an action, given the current observation.
        
        :param observation: the agent's current observation of the state of the world
        :return: the agent's action
        """
        # Get action probabilities using epsilon-greedy strategy
        action_probs = self.determine_action_probabilities(observation)
        
        action = numpy.random.choice(self.actions_n, p=action_probs)
        
        return action
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
        Updates the agent's Q-table using Q-learning update rule

        :param observation: The observation *before* the action
        :param action: The action that has been executed
        :param reward: The reward the action has yielded
        :param new_observation: The observation *after* the action
        :return:
        """
        # Get discretized states
        state = reshape_obs(observation)
        next_state = reshape_obs(new_observation)
        
        # update
        next_action = numpy.argmax(self.q[next_state])
        td_target = reward + self.gamma * self.q[next_state][next_action]
        td_delta = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_delta


        # decrease the epsilon by the decay vlaue
        self.step_counter += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
