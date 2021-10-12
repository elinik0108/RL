from OthelloEvaluator import OthelloEvaluator


class CountingEvaluator(OthelloEvaluator):
    def evaluate(self, othello_position):
        black_squares = 0
        white_squares = 0
        for row in othello_position.board:
            for item in row:
                if item == 'W':
                    white_squares += 1
                if item == 'B':
                    black_squares += 1
        return white_squares - black_squares
