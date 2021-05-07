""" Testing AB test functions for proportions

    testing whether base python functions are equivalent to statsmodels ones

"""
import numpy as np
import classical.proportions as proportions
import classical.workout.proportions as workout

np.random.seed(0)


def test_prop_conf_interval():
    sample1 = np.random.choice(2, 100, p=[0.5, 0.5])
    workout_ci = workout.one_prop_conf_interval(sample1)
    actual_ci = proportions.one_prop_conf_interval(sample1)
    assert workout_ci == actual_ci


def test_prop_hypothesis():
    sample1 = np.random.choice(2, 10, p=[0.55, 0.45])
    null_value = 0.40
    workout_ci = workout.one_prop_hypothesis(sample1, null_value)
    actual_ci = workout.one_prop_hypothesis(sample1, null_value)
    assert workout_ci == actual_ci


def test_props_hypothesis():
    sample1 = np.random.choice(2, 10, p=[0.5, 0.5])
    sample2 = np.random.choice(2, 10, p=[0.55, 0.45])
    workout_result = workout.two_props_hypothesis(sample1, sample2)
    actual_result = proportions.two_props_hypothesis(sample1, sample2)

    assert workout_result == actual_result


def test_props_conf_interval():
    sample1 = np.random.choice(2, 10, p=[0.5, 0.5])
    sample2 = np.random.choice(2, 10, p=[0.55, 0.45])
    conf = 0.95
    base_ci = workout.two_props_diff_conf_interval(sample1, sample2,
                                                   conf_level=conf)
    stats_ci = proportions.two_props_diff_conf_interval(sample1, sample2,
                                                        conf_level=conf)

    assert base_ci == stats_ci
