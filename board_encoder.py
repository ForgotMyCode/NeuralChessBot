import chess
import numpy as np

import meta_extractor

def encode_piece_type(piece_type: chess.PieceType) -> int:
    if piece_type == chess.PAWN:
        return 0
    if piece_type == chess.BISHOP:
        return 1
    if piece_type == chess.KNIGHT:
        return 2
    if piece_type == chess.ROOK:
        return 3
    if piece_type == chess.QUEEN:
        return 4
    if piece_type == chess.KING:
        return 5
    raise Exception("Invalid piece type " + str(piece_type))
    
def encode_fen_piece(piece: str) -> int:
    piece = piece.lower()
    if piece == 'p':
        return 0
    if piece == 'b':
        return 1
    if piece == 'n':
        return 2
    if piece == 'r':
        return 3
    if piece == 'q':
        return 4
    if piece == 'k':
        return 5
    raise Exception("Invalid piece type " + str(piece))

def encode_board(board: chess.Board, dtype='b'):
    encoded_board = np.zeros(shape=(8, 8, 7), dtype=dtype)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        if piece is None:
            encoded_board[7-chess.square_rank(square), chess.square_file(square), 6] = 1
            continue

        piece_type_index = encode_piece_type(piece.piece_type)
        encoded_board[7-chess.square_rank(square), chess.square_file(square), piece_type_index] = (1 if piece.color == chess.WHITE else -1)

    return encoded_board

def encode_board_meta(board: chess.Board, dtype='float32'):
    white_score, black_score = meta_extractor.count_piece_scores(board)
    return np.array([
        (1 if board.turn == chess.WHITE else -1),
        board.has_legal_en_passant(),
        board.has_kingside_castling_rights(chess.WHITE), 
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK), 
        board.has_queenside_castling_rights(chess.BLACK),
        white_score - black_score
    ], dtype=dtype)

def encode_move(row_from: int, col_from: int, row_to: int, col_to: int) -> int:
    return row_from + 8*col_from + 64*row_to + 512*col_to

def decode_move(encoded_position: int):
    col_to = encoded_position // 512
    encoded_position -= 512 * col_to
    row_to = encoded_position // 64
    encoded_position -= 64 * row_to
    col_from = encoded_position // 8
    encoded_position -= 8 * col_from
    row_from = encoded_position
    return (row_from, col_from, row_to, col_to)

def move_to_fen(row_from: int, row_to: int, col_from: int, col_to: int) -> str:
    return (chr(col_from + ord('a'))) + \
        str(8 - row_from) + \
        (chr(col_to) + ord('a')) + \
        str(8 - row_to)
