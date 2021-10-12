class OthelloAction(object):
    """
      This class represents a 'move' in a game.
      The move is simply represented by two integers: the row and the column where the player puts the marker and a
      boolean to mark if it is a pass move or not.
      In addition, the OthelloAction has a field where the estimated value of the move can be stored during
      computations.

      Author: Ola Ringdahl
    """

    def __init__(self, row, col, is_pass_move=False):
        """
        Creates a new OthelloAction for (row, col) with value 0.
        :param row: Row
        :param col: Column
        :param is_pass_move: True if it is a pass move
        """
        self.row = row
        self.col = col
        self.is_pass_move = is_pass_move
        self.value = 0

    def print_move(self):
        """
        Prints the move on the format (3,6) or Pass
        :return: Nothing
        """
        if self.is_pass_move:
            print("pass")
        else:
            print("(" + str(self.row) + "," + str(self.col) + ")")
