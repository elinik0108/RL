from OthelloAlgorithm import OthelloAlgorithm
from CountingEvaluator import CountingEvaluator
from OthelloAction import OthelloAction


class AlphaBeta(OthelloAlgorithm):
    """
    This is where you implement the alpha-beta-algorithm. 
	See OthelloAlgorithm for details
	
    Author:
    """
	DefaultDepth = 5 

    def __init__(self, othello_evaluator=CountingEvaluator(), depth=DefaultDepth):
        self.evaluator = othello_evaluator # change to your own evaluator
        self.search_depth = depth # you probably don't want to have a hard coded depth

    def set_evaluator(self, othello_evaluator, ):
        self.evaluator = othello_evaluator 

    def set_search_depth(self, depth):
        self.search_depth = depth 

    def evaluate(self, othello_position):
        # TODO: implement the alpha-beta algorithm