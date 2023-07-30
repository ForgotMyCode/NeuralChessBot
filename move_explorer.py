import chess

def get_legal_moves(board: chess.Board):
    return [legal_move for legal_move in board.legal_moves]

def expand_board(board: chess.Board, moves):
    next_boards = [board.copy() for _ in range(len(moves))]
    for i, move in enumerate(moves):
        next_boards[i].push(move)
    return next_boards

def expand_board_full(board: chess.Board):
    return expand_board(board, get_legal_moves(board))