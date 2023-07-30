import numpy as np
import tensorflow as tf

import batch_maker
import board_encoder
import chess
import chess.engine
import data_transformer
import sampling_distribution
import move_explorer

def evaluate_encoded_positions(model: tf.keras.Model, x):
    with tf.device("/device:GPU:0"):
        return np.ndarray.flatten(model.predict(x, verbose=0))

def evaluate_boards(model, boards):
    return evaluate_encoded_positions(model, batch_maker.make_boards_batch(boards))

def select_best_position(model, boards):
    evaluated_boards = evaluate_boards(model, boards)
    best_board_idx = np.argmax(evaluated_boards)
    return best_board_idx, evaluated_boards[best_board_idx]

def select_worst_position(model, boards):
    evaluated_boards = evaluate_boards(model, boards)
    best_board_idx = np.argmin(evaluated_boards)
    return best_board_idx, evaluated_boards[best_board_idx]

def predict_moves_probability_distribution(move_predicting_model: tf.keras.Model, board: chess.Board, moves) -> np.array:
    encoded_board = batch_maker.make_boards_batch([board])
    with tf.device("/device:GPU:0"):
        prediction = move_predicting_model.predict(encoded_board, verbose=0)
    encoded_moves = [data_transformer.encode_fen_move(str(move)) for move in moves]
    return prediction[0, encoded_moves]

def predict_moves_probability_distributions(move_predicting_model: tf.keras.Model, boards, movesLists) -> np.array:
    encoded_boards = batch_maker.make_boards_batch(boards)
    with tf.device("/device:GPU:0"):
        prediction = move_predicting_model.predict(encoded_boards, verbose=0)
    encoded_moves = [[data_transformer.encode_fen_move(str(move)) for move in moves] for moves in movesLists]
    return [prediction[i, encoded_moves[i]] for i in range(len(encoded_moves))]

def predict_moves_probability_distributions_experimental(value_predicting_model: tf.keras.Model, boards, lists_of_moves, is_current_player_max: bool) -> np.array:
    flat_boards, size_info = batch_maker.flatten([move_explorer.expand_board(board, moves) for board, moves in zip(boards, lists_of_moves)])
    encoded_boards = batch_maker.make_boards_batch(flat_boards)
    encoded_metas = batch_maker.make_meta_batch(flat_boards)
    with tf.device("/device:GPU:0"):
        prediction = value_predicting_model.predict([encoded_boards, encoded_metas], verbose=0).flatten()
        if is_current_player_max:
            prediction = sampling_distribution.resample_from_s2_up(prediction)
        else:
            prediction = sampling_distribution.resample_from_s2_down(prediction)
        return batch_maker.unflatten(prediction, size_info)

def predict_position_value_from_encoded(value_predicting_model, encoded_board, encoded_meta):
    with tf.device("/device:GPU:0"):
        return value_predicting_model.predict([encoded_board, encoded_meta], verbose=0)[0]
    
def predict_positions_values_from_encoded(value_predicting_model, encoded_boards, encoded_metas):
    with tf.device("/device:GPU:0"):
        return value_predicting_model.predict([encoded_boards, encoded_metas])

def predict_position_value(value_predicting_model, board) -> float:
    encoded_board = batch_maker.make_boards_batch([board])
    encoded_meta = batch_maker.make_meta_batch([board])
    prediction = predict_position_value_from_encoded(value_predicting_model, encoded_board, encoded_meta)
    return prediction

def predict_positions_values(value_predicting_model, boards) -> np.array:
    encoded_boards = batch_maker.make_boards_batch(boards)
    encoded_metas = batch_maker.make_meta_batch(boards)
    predictions = predict_positions_values_from_encoded(value_predicting_model, encoded_boards, encoded_metas)
    return predictions.flatten()

def predict_using_engine(engine: chess.engine.SimpleEngine, board: chess.Board):
    result = engine.play(board, chess.engine.Limit(depth=0), info=chess.engine.INFO_SCORE)
    return result.move, result.info["score"].white().wdl().expectation()*2.0 - 1.0