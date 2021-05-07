""" Testing AB test functions for means

    testing whether base python functions are equivalent to statsmodels ones

"""
import numpy as np
import classical.means as means
import classical.workout.means as workout


np.random.seed(0)


def test_mean_conf_interval():
    sample1 = np.random.normal(50, 1, 10)
    workout_ci = workout.one_mean_conf_interval(sample1)
    actual_ci = means.one_mean_conf_interval(sample1)
    assert workout_ci == actual_ci


def test_means_hypothesis():
    sample1 = np.random.normal(50, 1, 10)
    sample2 = np.random.normal(51, 1, 10)
    workout_result = workout.two_means_hypothesis(sample1, sample2,
                                                  pooled=True)
    actual_result = means.two_means_hypothesis(sample1, sample2,
                                               pooled=True)

    assert __apply_round(workout_result) == __apply_round(actual_result)


def test_means_conf_interval():
    sample1 = np.random.normal(50, 1, 10)
    sample2 = np.random.normal(51, 1, 10)
    conf = 0.95
    base_ci = workout.two_means_diff_conf_interval(sample1, sample2,
                                                   conf_level=conf,
                                                   pooled=True)
    stats_ci = means.two_means_diff_conf_interval(sample1, sample2,
                                                  conf_level=conf, pooled=True)

    assert base_ci == stats_ci


def __apply_round(results, ndigits=6):
    return [round(val, ndigits) for val in results]
