import chess

def get_piece_value(piece_type: chess.PieceType) -> float:
    if piece_type == chess.PAWN:
        return 1.0
    if piece_type == chess.BISHOP:
        return 3.2
    if piece_type == chess.KNIGHT:
        return 3.0
    if piece_type == chess.ROOK:
        return 5.0
    if piece_type == chess.QUEEN:
        return 9.0
    if piece_type == chess.KING:
        return 0.0
    raise Exception("Invalid piece type " + str(piece_type))

def get_fen_piece_value(piece: str) -> float:
    piece = piece.lower()
    if piece == 'p':
        return 1.0
    if piece == 'b':
        return 3.2
    if piece == 'n':
        return 3.0
    if piece == 'r':
        return 5.0
    if piece == 'q':
        return 9.0
    if piece == 'k':
        return 0.0
    raise Exception("Invalid piece type " + str(piece))

def count_piece_scores(board: chess.Board):
    white_score = 0.0
    black_score = 0.0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        if piece is None:
            continue
        
        piece_value = get_piece_value(piece.piece_type)
        
        if piece.color == chess.WHITE:
            white_score += piece_value
        else:
            black_score += piece_value
            
    return white_score, black_score