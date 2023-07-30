import chess
import math
import time
import numpy as np
from collections import deque

import board_encoder
import meta_extractor
import move_explorer
import predictor
import sampling_distribution
import time
import heapq

UB = 10000.0

def internal_negamax(model, board: chess.Board, alpha, beta, remaining_depth, critical_time, expansions, expansions_limit):    
    # check if game is over
    outcome = board.outcome()
    if outcome is not None:
        if board.turn == outcome.winner:
            return UB
        if (not board.turn) == outcome.winner:
            return -UB
        return 0.0

    moves = move_explorer.get_legal_moves(board)

    if remaining_depth <= 0 or expansions[0] > expansions_limit or time.time() > critical_time:
        return predictor.select_worst_position(model, move_explorer.expand_board(board, moves))[1]
    
    expansions[0] += len(moves)

    evaluation = -math.inf

    for move in moves:
        board.push(move)
        evaluation = max(evaluation, -internal_negamax(model, board, -beta, -alpha, remaining_depth-1, critical_time, expansions, expansions_limit))
        board.pop()
        alpha = max(alpha, evaluation)

        if alpha >= beta:
            break

    return evaluation

def aggressive_minimax(model, board: chess.Board, max_move_expansion, remaining_depth, critical_time, expansions, expansions_limit):
    # check if game is over
    outcome = board.outcome()
    if outcome is not None:
        if board.turn == outcome.winner:
            return UB
        if (not board.turn) == outcome.winner:
            return -UB
        return 0.0

    moves = move_explorer.get_legal_moves(board)

    if remaining_depth <= 0 or expansions[0] > expansions_limit or time.time() > critical_time:
        return predictor.select_worst_position(model, move_explorer.expand_board(board, moves))[1]

    future_boards = move_explorer.expand_board(board, moves)
    evaluations = sorted([(evaluation, i) for i, evaluation in enumerate(predictor.evaluate_boards(model, future_boards))])

    move_cutoff = min(len(evaluations), max_move_expansion)
    expansions[0] += len(moves)

    best_evaluation = math.inf

    for _, i_move in evaluations[:move_cutoff]:
        evaluation = aggressive_minimax(model, future_boards[i_move], max_move_expansion, remaining_depth - 1, critical_time, expansions, expansions_limit)
        best_evaluation = min(best_evaluation, evaluation)

    return best_evaluation

def sampling_search(value_model, policy_model, board: chess.Board, depth_limit, sample_limit):
    def explore(value_model, policy_model, board: chess.Board, remaining_depth, samples):
        outcome = board.outcome()        
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                return 1.0
            if outcome.winner == chess.BLACK:
                return -1.0
            return 0.0
        
        if remaining_depth <= 0:
            return predictor.predict_position_value(value_model, board)

        moves = move_explorer.get_legal_moves(board)
        distribution = predictor.predict_moves_probability_distribution(policy_model, board, moves)
        move_samples = sampling_distribution.noisy_sample_from_distribution(distribution, samples)
        move_idxs, move_counts = np.unique(move_samples, return_counts=True)
        value = 0.0

        for move_idx, move_count in zip(move_idxs, move_counts):
            if move_count <= 0:
                continue
            board.push(moves[move_idx])
            value += explore(value_model, policy_model, board, remaining_depth - 1, move_count)
            board.pop()

        return value / float(samples)

    return explore(value_model, policy_model, board, depth_limit, sample_limit)

class CalmSearchState:
    def __init__(self, is_max_player):
        self.children = []
        self.value = None
        self.is_max_player = is_max_player
        self.board = None

    def recalculate_value(self):
        if self.value is not None:
            return self.value
        
        if self.is_max_player:
            self.value = max(child.recalculate_value() for child in self.children)
        else:
            self.value = min(child.recalculate_value() for child in self.children)
        return self.value

    def calm_search(self, board: chess.Board, depth, leaf_collector):
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                self.value = (1 + 1.0 / depth) * 999
            elif outcome.winner == chess.BLACK:
                self.value = -(1 + 1.0 / depth) * 999
            else:
                self.value = 0.0
            return
        
        is_calm_state = True

        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                is_calm_state = False
                child = CalmSearchState(not self.is_max_player)
                child.calm_search(board, depth + 1, leaf_collector)
                board.pop()
                self.children.append(child)
            else:
                board.push(move)
                if board.is_check():
                    is_calm_state = False
                    child = CalmSearchState(not self.is_max_player)
                    child.calm_search(board, depth + 1, leaf_collector)
                    self.children.append(child)
                board.pop()

        if is_calm_state:
            leaf_collector.append(self)
            self.board = board.copy()

def best_move(value_model, policy_model, board: chess.Board, moves, sampling_limit = 100, computing_power = 10000000):
    class InterestingState:
        def __init__(self, board: chess.Board, is_capture: bool, was_check: bool, depth: int):
            self.board = board.copy()
            white, black = meta_extractor.count_piece_scores(board)
            self.basic_interest = 10 + (abs(white - black) / 6.0)
            if board.is_check():
                self.basic_interest += 10
            if is_capture:
                self.basic_interest += 10
            if was_check:
                self.basic_interest += 10
            self.basic_interest *=  (0.55 ** depth)
            self.children = None
            self.value = (white - black) / 2.0
            self.depth = depth
            self.do_not_explore = False
            self.max_explored_depth = 0

            outcome = board.outcome()
            if outcome is not None:
                self.do_not_explore = True
                if outcome.winner == chess.WHITE:
                    self.value = (1 + 1.0 / depth) * 999
                elif outcome.winner == chess.BLACK:
                    self.value = -(1 + 1.0 / depth) * 999
                else:
                    self.value = 0.0
                return

        def recalculate_value(self):
            if self.children is None:
                return self.value
            
            if self.board.turn == chess.WHITE:
                self.value = max([child.recalculate_value() for child in self.children])
            else:
                self.value = min([child.recalculate_value() for child in self.children])

            self.max_explored_depth = 1 + max([child.max_explored_depth for child in self.children])

            return self.value
            
        def explore(self, heap):
            children = []

            for move in move_explorer.get_legal_moves(self.board):
                is_capture = self.board.is_capture(move)
                was_check = self.board.is_check()
                self.board.push(move)
                child = InterestingState(self.board, is_capture, was_check, self.depth + 1)
                heapq.heappush(heap, child)
                children.append(child)
                self.board.pop()

            if len(children) > 0:
                self.children = children

        def __lt__(self, other) -> bool:
            return self.basic_interest > other.basic_interest

    root_states = []

    for move in moves:
        is_capture = board.is_capture(move)        
        was_check = board.is_check()
        board.push(move)
        root_states.append(InterestingState(board, is_capture, was_check, 1))
        board.pop()

    heap = root_states.copy()
    heapq.heapify(heap)

    while len(heap) < computing_power and len(heap) > 0:
        state = heapq.heappop(heap)
        state.explore(heap)

    currently_best_move = 0
    currently_best_evaluation = (-math.inf) if board.turn == chess.WHITE else math.inf

    print("Heuristically evaluating", len(heap), "boards ...")
    t1 = time.perf_counter()
    values = predictor.predict_positions_values(value_model, [state.board for state in heap])
    t2 = time.perf_counter()

    print("Spent", t2 - t1, "sec in predictor")

    for state, value in zip(heap, values):
        state.value += value
    
    currently_best_move = 0
    currently_best_evaluation = (-math.inf) if board.turn == chess.WHITE else math.inf

    for i, move in enumerate(moves):
        evaluation = root_states[i].recalculate_value()

        print("[DEBUG]\t " + str(move) + " evaluated as " + str(evaluation), "(Interest:", root_states[i].basic_interest, "MaxDepth = ", root_states[i].max_explored_depth, ")")

        if board.turn == chess.WHITE:
            if evaluation > currently_best_evaluation:
                currently_best_move = i
                currently_best_evaluation = evaluation
        else:
            if evaluation < currently_best_evaluation:
                currently_best_move = i
                currently_best_evaluation = evaluation

    return currently_best_move, currently_best_evaluation

def best_move_v2(value_model, policy_model, board: chess.Board, moves, sampling_limit = 100, computing_power = 10000000):
    currently_best_move = 0
    currently_best_evaluation = (-math.inf) if board.turn == chess.WHITE else math.inf
    
    roots = []
    leaves = []

    for move in moves:
        board.push(move)
        state = CalmSearchState(board.turn == chess.WHITE)
        state.calm_search(board, 0, leaves)
        roots.append(state)
        board.pop()

    print("Heuristically evaluating", len(leaves), "boards ...")
    t1 = time.perf_counter()
    values = predictor.predict_positions_values(value_model, [state.board for state in leaves])
    t2 = time.perf_counter()
    piece_counts = [meta_extractor.count_piece_scores(state.board) for state in leaves]
    t3 = time.perf_counter()

    print("Spent", t2 - t1, "sec in predictor")
    print("Spent", t3 - t2, "sec counting pieces")

    for state, value, piece_count in zip(leaves, values, piece_counts):
        state.value = piece_count[0] - piece_count[1] + value
    
    currently_best_move = 0
    currently_best_evaluation = (-math.inf) if board.turn == chess.WHITE else math.inf

    for i, move in enumerate(moves):
        evaluation = roots[i].recalculate_value()

        print("[DEBUG]\t " + str(move) + " evaluated as " + str(evaluation), "(", roots[i].value, ")")

        if board.turn == chess.WHITE:
            if evaluation > currently_best_evaluation:
                currently_best_move = i
                currently_best_evaluation = evaluation
        else:
            if evaluation < currently_best_evaluation:
                currently_best_move = i
                currently_best_evaluation = evaluation

    return currently_best_move, currently_best_evaluation

def best_move_v1(value_model, policy_model, board: chess.Board, moves, sampling_limit = 100, computing_power = 10000000):
    def advance_board(board: chess.Board, move: chess.Move):
        board_clone = board.copy()
        board_clone.push(move)
        return board_clone

    depth_limit = max(1, round(math.log(computing_power, max(2, len(moves)))))
    print("[DEBUG]\t", "Launching Deep search, estimated to", depth_limit, '...')
    currently_best_move = 0
    currently_best_evaluation = (-math.inf) if board.turn == chess.WHITE else math.inf

    class State:
        def __init__(self, board, is_max_player, depth):
            self.board = board
            self.depth = depth
            #self.sample_limit = sample_limit
            #self.parent = parent
            self.value = None
            self.children = None
            self.is_max_player = is_max_player

        def recalculate_value(self):
            if self.value is not None:
                return self.value
            
            if self.is_max_player:
                self.value = max([child.recalculate_value() for child in self.children])
            else:
                self.value = min([child.recalculate_value() for child in self.children])
            return self.value

        def accumulate_child_value(self):
            # deprecated
            if self.children is not None:
                if self.is_max_player:
                    self.value = max([child.value for child in self.children])
                else:
                    self.value = min([child.value for child in self.children])
                
    search_seq = [State(advance_board(board, move), board.turn != chess.WHITE, 1) for move in moves]

    q = deque(search_seq)

    used_power = 0

    while used_power < computing_power and len(q) > 0:
        used_power += 1

        node = q[0]
        q.popleft()

        outcome = node.board.outcome()
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                node.value = (1 + 1.0 / node.depth) * 999
            elif outcome.winner == chess.BLACK:
                node.value = -(1 + 1.0 / node.depth) * 999
            else:
                node.value = 0.0
            continue

        node.children = [State(advance_board(node.board, move), not node.is_max_player, node.depth + 1) for move in move_explorer.get_legal_moves(node.board)]
        q.extend(node.children)

    if len(q) > 0:
        print("Min search depth = ", q[0].depth)

    print("Heuristically evaluating", len(q), "boards ...")
    t1 = time.perf_counter()
    values = predictor.predict_positions_values(value_model, [state.board for state in q])
    t2 = time.perf_counter()
    piece_counts = [meta_extractor.count_piece_scores(state.board) for state in q]
    t3 = time.perf_counter()

    print("Spent", t2 - t1, "sec in predictor")
    print("Spent", t3 - t2, "sec counting pieces")

    for i, state in enumerate(q):
        state.value = piece_counts[i][0] - piece_counts[i][1] + values[i]
    
    currently_best_move = 0
    currently_best_evaluation = (-math.inf) if board.turn == chess.WHITE else math.inf

    for i, move in enumerate(moves):
        evaluation = search_seq[i].recalculate_value()

        print("[DEBUG]\t " + str(move) + " evaluated as " + str(evaluation), "(", search_seq[i].value, ")")

        if board.turn == chess.WHITE:
            if evaluation > currently_best_evaluation:
                currently_best_move = i
                currently_best_evaluation = evaluation
        else:
            if evaluation < currently_best_evaluation:
                currently_best_move = i
                currently_best_evaluation = evaluation

    return currently_best_move, currently_best_evaluation
        