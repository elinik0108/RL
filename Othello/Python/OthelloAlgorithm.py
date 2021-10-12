from abc import ABC, abstractmethod


class OthelloAlgorithm(ABC):
    """
    This interface (abstract class) defines the mandatory methods for game playing algorithms, i.e., algorithms that
    take an OthelloAlgorithm and return a suggested move for the player who has the move.

    The algorithm only defines the search method. The heuristic evaluation of positions is given by an
    OthelloEvaluator which is given to the algorithm.

    Author: Ola Ringdahl
    """

    @abstractmethod
    def set_evaluator(self, othello_evaluator):
        """
        Sets an OthelloEvaluator the algorithm is to use for heuristic evaluation.
        :param othello_evaluator: the OthelloEvaluator
        :return: Nothing
        """
        pass

    @abstractmethod
    def evaluate(self, othello_position):
        """
        Returns the OthelloAction the algorithm considers to be the best move given an OthelloPosition
        :param othello_position: The OthelloPosition to evaluate
        :return: The move to make as an OthelloAction
        """
        pass

    @abstractmethod
    def set_search_depth(self, depth):
        """
        Sets the maximum search depth of the algorithm.
        :param depth: maximum search depth
        :return: Nothing
        """
        pass
