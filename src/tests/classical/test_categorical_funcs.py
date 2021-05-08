import numpy as np
import classical.workout.categorical as workout
import classical.categorical as categorical


def test_one_categorical_hypothesis():
    counts = np.array([1920, 347, 19, 84])
    nobs = np.array([2392, 2878, 2405, 2877])
    workout_result = workout.one_categorical_hypothesis(counts, nobs)
    actual_result = categorical.one_categorical_hypothesis(counts, nobs)
    assert workout_result == actual_result


def test_two_categorical_hypothesis():
    observed = np.array([[10, 10, 20], [20, 20, 20]])
    print(observed.shape)
    workout_result = workout.two_categorical_hypothesis(observed)
    actual_result = categorical.two_categorical_hypothesis(observed)
    assert workout_result == actual_result
