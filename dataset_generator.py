import chess
import chess.engine
import csv
import random

import move_explorer
import predictor

def generate_random_positions(n, min_moves = 5, max_moves = 40):
    positions = []
    for _ in range(n):
        board = chess.Board()
        moves = random.randint(min_moves, max_moves)
        for _ in range(moves):
            legal_moves = move_explorer.get_legal_moves(board)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
        positions.append(board)
    return positions

def generate_dataset(n, output_path, min_moves = 5, max_moves = 35):
    engine = chess.engine.SimpleEngine.popen_uci('D:/downloads/stockfish-windows-x86_64-modern.exe')
    f = open(output_path, "w", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["FEN", "Evaluation", "Move"])

    for i in range(n):
        if i % (n // 100) == 0:
            print((i * 100) // n, '%')
        board = chess.Board()
        moves = random.randint(min_moves, max_moves)
        for _ in range(moves):
            legal_moves = move_explorer.get_legal_moves(board)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
        best_move, value = predictor.predict_using_engine(engine, board)
        csv_writer.writerow([board.fen(), value, str(best_move)])
        
    engine.quit()
    f.close()
