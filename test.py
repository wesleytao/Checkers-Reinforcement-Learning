from Board import *
import random

def main():
    this_board = Board()
    for i in range(5):
        this_board.print_board()
        moves = this_board.get_possible_next_moves()
        print("----current player {}-------".format(this_board.player_turn))
        print("-----available moves-------")
        print(moves)
        print(this_board.current_state())
        this_move = random.choice(moves)
        print("step  {} pick {}".format(i, this_move))
        this_board.make_move(this_move)


if __name__ == "__main__":
    main()
