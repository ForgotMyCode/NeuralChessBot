import numpy as np

import batch_maker
import predictor

def test_who_wins(value_model, x, y):
    def evaluate_test(x, y):
        if y > 0 and x > 0:
            return True
        if y < 0 and x < 0:
            return True
        if y == 0 and abs(x) < 0.25:
            return True
        return False

    results = [
        evaluate_test(
            predictor.predict_position_value_from_encoded(
                value_model,
                batch_maker.make_batch([x[0][i, :]]),
                batch_maker.make_batch([x[1][i, :]])
            ),
            y[i]
		) for i in range(len(x[0]))
    ]
    correct = sum(results)
    total = len(results)
    print("Test result: ", correct, '/', total, '(', float(correct * 100) / float(total), '%)')