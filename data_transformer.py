import chess
import numpy as np
import board_encoder
import meta_extractor

def transform_stockfish_evaluation(x):
    UB = 10000
    if x.startswith('#'):
        m = float(x[1:])
        return (-UB if x[2] == '-' else UB) - m
    return float(x)

def transform_stockfish_evaluation_to_who_is_winning(x: str) -> float:
    if x == '0':
        return 0.0
    return -1.0 if x.startswith('-') or x.startswith('#-') else 1.0

def transform_fen_to_board(fen: str) -> chess.Board:
    return chess.Board(fen)

def has_fen_piece_players_color(piece: str, is_white_player):
    return (is_white_player and piece.isupper()) or ((not is_white_player) and piece.islower())

def is_fen_piece_white(piece: str) -> bool:
    return piece.isupper()

def safe_float(s: str) -> float:
    return max(-0.95, min(0.95, float(s)))

def transform_fen_to_np_encoded_board(fen: str, dtype='b'):
    board = np.zeros(shape=(8, 8, 7), dtype=dtype)
    meta = np.zeros(shape=(7,), dtype=dtype)

    parts = fen.split(' ')

    for i_row, row in enumerate(parts[0].split('/')):
        i_col = 0

        for c in row:
            if c.isdigit():
                offset = ord(c) - ord('0')
                for i in range(i_col, i_col + offset):
                    board[i_row, i, 6] = 1
                i_col += offset
                continue
            index = board_encoder.encode_fen_piece(c)
            whiteness_multiplier = (1 if is_fen_piece_white(c) else -1)
            board[i_row, i_col, index] = whiteness_multiplier
            piece_value = meta_extractor.get_fen_piece_value(c)
            meta[6] += piece_value * whiteness_multiplier
            i_col += 1

    is_white_player_turn = parts[1] == 'w'

    meta[0] = (1 if is_white_player_turn else -1)
    meta[1] = int(parts[3] != '-')
    meta[2] = 'K' in parts[2]
    meta[3] = 'Q' in parts[2]
    meta[4] = 'k' in parts[2]
    meta[5] = 'q' in parts[2]

    return board, meta

def encode_fen_move(fen_move: str) -> int:
    col_from = ord(fen_move[0]) - ord('a')
    row_from = 8 - int(fen_move[1])
    col_to = ord(fen_move[2]) - ord('a')
    row_to = 8 - int(fen_move[3])
    return board_encoder.encode_move(row_from, col_from, row_to, col_to)

def one_hot_encode(value, size, dtype = 'b') -> np.array:
    ret = np.zeros(shape=(size,), dtype=dtype)
    ret[value] = 1
    return ret