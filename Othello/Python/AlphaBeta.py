from OthelloAlgorithm import OthelloAlgorithm
from CountingEvaluator import CountingEvaluator
from OthelloAction import OthelloAction


class AlphaBeta(OthelloAlgorithm):
    """
    This is where you implement the alpha-beta algorithm. 
	See OthelloAlgorithm for details
	
    Author:
    """
	DefaultDepth = 5 

    def __init__(self, othello_evaluator=CountingEvaluator(), depth=DefaultDepth):
        self.evaluator = othello_evaluator 
        self.search_depth = depth 

    def set_evaluator(self, othello_evaluator, ):
        self.evaluator = othello_evaluator # change to your own evaluator

    def set_search_depth(self, depth):
        self.search_depth = depth # use iterative deepening search to decide depth

    def evaluate(self, othello_position):
        # TODO: implement the alpha-beta algorithm