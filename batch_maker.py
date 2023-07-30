import numpy as np
import board_encoder

def make_boards_batch(boards):
    encoded_boards = []

    for board in boards:
        encoded_board = board_encoder.encode_board(board)
        encoded_boards.append(encoded_board)

    return np.stack(encoded_boards)

def make_meta_batch(boards):
    metas = []

    for board in boards:
        meta = board_encoder.encode_board_meta(board)
        metas.append(meta)

    return np.stack(metas)

def make_batch(collection):
    return np.stack(collection)

def make_batch_from_items_at(collection, key):
    return np.stack([element[key] for element in collection])

def flatten(list_of_lists):
    flat_list = []
    size_info = []

    for l in list_of_lists:
        start = len(flat_list)
        flat_list += l
        end = len(flat_list)
        size_info.append((start, end))

    return flat_list, size_info

def unflatten(flat_list, size_info):
    result = []

    for start, end in size_info:
        result.append(flat_list[start:end])

    return result

