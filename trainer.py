import datetime
import pandas as pd
import tensorflow as tf
import random

import batch_maker
import data_transformer
import model_backup

def take_csv_sample(path, samples):
    records = sum(1 for _ in open(path)) - 1
    skip = sorted(random.sample(range(1, records + 1), records - samples))
    return pd.read_csv(path, skiprows=skip, dtype=str)

def read_csv(path):
    return pd.read_csv(path, dtype=str)

def load_data_type_001(path):
    pass
    # df = pd.read_csv(path)
    # y = df['Evaluation'].apply(data_transformer.transform_stockfish_evaluation)
    # x = df['FEN'].apply(data_transformer.transform_fen_to_np_encoded_board)
    # # flip the value if current player is black because engine evaluations are not relative
    # for i, encoded_board in enumerate(x):
    #     if not encoded_board[2]:
    #         y[i] *= -1.0
    # x = batch_maker.make_batches_from_encoded(x)
    # return x, y

def load_data_type_002(path):
    pass
    # df = pd.read_csv(path)
    # y = df['Evaluation'].apply(lambda x: x.replace('\ufeff', ''))
    # y = y.apply(data_transformer.transform_stockfish_evaluation)
    # x = df['FEN'].apply(data_transformer.transform_fen_to_np_encoded_board)
    # # flip the value if current player is black because engine evaluations are not relative
    # for i, encoded_board in enumerate(x):
    #     if not encoded_board[2]:
    #         y[i] *= -1.0
    # x = batch_maker.make_batches_from_encoded(x)
    # return x, y

def load_data_type_003(path, samples):
    df = take_csv_sample(path, samples).dropna() if samples is not None else read_csv(path).dropna()
    y = df['Evaluation'].apply(data_transformer.transform_stockfish_evaluation_to_who_is_winning)
    x = df['FEN'].apply(data_transformer.transform_fen_to_np_encoded_board)    
    return (batch_maker.make_batch_from_items_at(x, 0), batch_maker.make_batch_from_items_at(x, 1)), y

def load_data_type_004(path, samples):
    df = take_csv_sample(path, samples).dropna() if samples is not None else read_csv(path).dropna()
    x = df['FEN'].apply(data_transformer.transform_fen_to_np_encoded_board)
    y = df['Move'].apply(lambda move: data_transformer.one_hot_encode(data_transformer.encode_fen_move(move), 64*64, 'float32'))
    return batch_maker.make_batch_from_items_at(x, 0), batch_maker.make_batch(y)

def load_data_type_005(path, samples = None):
    df = take_csv_sample(path, samples).dropna() if samples is not None else read_csv(path).dropna()
    y = df['Evaluation'].apply(data_transformer.safe_float)
    x = df['FEN'].apply(data_transformer.transform_fen_to_np_encoded_board)    
    return (batch_maker.make_batch_from_items_at(x, 0), batch_maker.make_batch_from_items_at(x, 1)), y

def internal_train(model, x, y, epochs, validation_split, batch_size):
    with tf.device("/device:GPU:0"):
        history = model.fit(
            x, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='loss'), 
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=0.001)
            ]
        )
        return history
    
def train(model, x, y, epochs = 1, validation_split = 0.1, batch_size = 64, autosave=True):
    history = internal_train(model, x, y, epochs, validation_split, batch_size)
    if autosave:
        model_backup.autosave(model)
    return history