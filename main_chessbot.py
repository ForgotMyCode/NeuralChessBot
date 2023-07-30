import chess

import dataset_generator
import deep_search
import model_backup
import move_explorer
import positional_model
import predictor
import self_train
import tester
import trainer

def info(message):
    print("[INFO]\t", message)

def main():
    should_exit = False
    model = None
    
    while not should_exit:
        #try:
            if model is None:
                user_input = input("""
There is currently no model loaded.
Use:
 • new [type]
   (Create new model)
 • load [path]
   (Load value and policy model from given path)
 • gendata [size] [path]
   (generate data)
""")
                args = user_input.split()

                if len(args) == 2 and args[0].lower() == 'new':
                    type = int(args[1])
                    info("Creating new model...")
                    model = None
                    if type == 1:
                        model = [positional_model.get_value_model_003(), positional_model.get_policy_model_003()]
                    else:
                        info("Invalid model type")
                        continue
                    info("Compiling model...")
                    positional_model.compile_regression_model(model[0])
                    positional_model.compile_experimental_classifier_model(model[1])
                    info("Done!")
                    continue
                
                if len(args) == 2 and args[0].lower() == 'load':
                    info("Loading model...")
                    model = model_backup.load_model(args[1])
                    info("Done!")
                    continue

                if len(args) == 3 and args[0].lower() == 'gendata':
                    size = int(args[1])
                    path = args[2]
                    info("Generating dataset...")
                    dataset_generator.generate_dataset(size, path)
                    info("Done!")
                    continue
            else:
                user_input = input("""
There is a model loaded.
Use:
 • unload
   (Unload the model WITHOUT SAVING)
 • save [model] [path]
   (Save the model (1 = value, 2 = policy) to given path)
 • datatrain [model] [type] [path] [epochs] [samples]
   (Train the model (1 = value, 2 = policy) using dataset)
 • bestmove [samples] [computing power] [fen position]
   (Estimate best move given a position in FEN format)
 • selftrain [games] [samples] [computing power]
   (Learn by playing games against itself)
 • valuetest [dataset] [samples]
   (Test value predicting model)
 • summary [model]
   (print model info)
""")
                if user_input.lower() == 'unload':
                    model = None
                    continue
                
                args = user_input.split() # TODO: not great, needs ""

                if len(args) == 2 and args[0].lower() == 'save':
                    info("Saving model to \"" + args[1] + "\"...")
                    model_backup.save_model(model[0], args[1])
                    info("Done!")
                    continue

                if(len(args) == 5 and args[0].lower() == 'datatrain'):
                    model_to_train = model
                    type = int(args[1])
                    path = args[2]
                    epochs = int(args[3])
                    samples = int(args[4]) if args[5].isdigit() else None

                    x = y = None

                    if type == 1:
                        info("Loading type 1 data")
                        x, y = trainer.load_data_type_001(path)
                    elif type == 2:
                        info("Loading type 2 data")
                        x, y = trainer.load_data_type_002(path)
                    elif type == 3:
                        info("Loading type 3 data")
                        x, y = trainer.load_data_type_003(path, samples)
                    elif type == 4:
                        info("Loading type 4 data")
                        x, y = trainer.load_data_type_004(path, samples)
                    elif type == 5:
                        info("Loading type 5 data")
                        x, y = trainer.load_data_type_005(path, samples)
                    else:
                        info("Invalid type")
                        continue

                    info("Training...")
                    history = trainer.train(model_to_train, x, y, int(epochs))
                    info(history)
                    info("Done!")
                    continue

                if user_input.startswith('bestmove'):
                    samples = int(args[1])
                    computing_power = int(args[2])
                    info("Building board...")
                    fen = ' '.join(args[3:])
                    board = chess.Board(fen)
                    info("Exploring moves...")
                    moves = move_explorer.get_legal_moves(board)
                    info("Evaluating boards...")
                    best_future_position_idx, value = deep_search.best_move(model, None, board, moves, samples, computing_power)
                    print(str(moves[best_future_position_idx]), value)
                    info("Done!")
                    continue

                if len(args) == 4 and args[0].lower() == 'selftrain':
                    games = int(args[1])
                    samples = int(args[2])
                    computing_power = int(args[3])

                    self_train.self_train(model[0], model[1], games, samples, computing_power)
                    continue

                if len(args) == 3 and args[0].lower() == 'valuetest':
                    path = args[1]
                    samples = int(args[2]) if args[2].isdigit() else None
                    x, y = trainer.load_data_type_003(path, samples)
                    tester.test_who_wins(model[0], x, y)
                    continue

                if len(args) == 2 and args[0].lower() == 'summary':
                    print(model)
                    i_model = int(args[1]) - 1
                    print(model[i_model].summary())
                    continue

        #except Exception as e:
        #    print(e)

    
if __name__ == '__main__':
    main()