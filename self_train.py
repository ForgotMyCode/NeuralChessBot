import chess
import numpy as np
import time

import batch_maker
import board_encoder
import data_transformer
import deep_search
import model_backup
import move_explorer
import trainer

def self_train(value_model, policy_model, games, samples, computing_power):
    for game in range(games):
        print("Playing game", (game + 1), '/', games, '...')
        board = chess.Board()
        encoded_boards = []
        encoded_metas = []
        next_encoded_moves = []
        
        while not board.is_game_over():
            moves = move_explorer.get_legal_moves(board)
            move, _ = deep_search.best_move(value_model, policy_model, board, moves, samples, computing_power)
            encoded_boards.append(board_encoder.encode_board(board))
            encoded_metas.append(board_encoder.encode_board_meta(board))
            next_encoded_moves.append(data_transformer.one_hot_encode(data_transformer.encode_fen_move(str(moves[move])), 64*64, 'float32'))
            board.push(moves[move])
        
        value = None

        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                value = np.array([1.0] * len(encoded_boards))
            elif outcome.winner == chess.BLACK:
                value = np.array([-1.0] * len(encoded_boards))
            else:
                value = np.array([0.0] * len(encoded_boards))
                
        encoded_boards = batch_maker.make_batch(encoded_boards)
        encoded_metas = batch_maker.make_batch(encoded_metas)
        next_encoded_moves = batch_maker.make_batch(next_encoded_moves)
                
        trainer.train(value_model, [encoded_boards, encoded_metas], value, 5, 0.0, 32, False)
        trainer.train(policy_model, encoded_boards, next_encoded_moves, 5, 0.0, 32, False)

    model_backup.autosave(value_model)
    time.sleep(2)
    model_backup.autosave(policy_model)
    
        