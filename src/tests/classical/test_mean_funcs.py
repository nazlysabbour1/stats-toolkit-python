""" Testing AB test functions for means

    testing whether base python functions are equivalent to statsmodels ones

"""
import numpy as np
import classical.means as means
import classical.workout.means as workout
import pytest

np.random.seed(0)


def test_mean_conf_interval():
    sample1 = np.random.normal(50, 1, 10)
    workout_ci = workout.one_mean_conf_interval(sample1)
    actual_ci = means.one_mean_conf_interval(sample1)
    assert workout_ci == pytest.approx(actual_ci)


def test_mean_hypothesis():
    sample = np.random.normal(50, 1.5, 10)
    null_value = 55
    workout_result = workout.one_mean_hypothesis(sample, null_value)
    actual_result = means.one_mean_hypothesis(sample, null_value)
    assert workout_result == pytest.approx(actual_result)


def test_two_means_hypothesis():
    sample1 = np.random.normal(50, 1, 10)
    sample2 = np.random.normal(51, 1, 10)
    workout_result = workout.two_means_hypothesis(sample1, sample2,
                                                  pooled=True)
    actual_result = means.two_means_hypothesis(sample1, sample2,
                                               pooled=True)

    assert workout_result == pytest.approx(actual_result)


def test_means_diff_conf_interval():
    sample1 = np.random.normal(50, 1, 10)
    sample2 = np.random.normal(51, 1, 10)
    conf = 0.95
    base_ci = workout.two_means_diff_conf_interval(sample1, sample2,
                                                   conf_level=conf,
                                                   pooled=True)
    stats_ci = means.two_means_diff_conf_interval(sample1, sample2,
                                                  conf_level=conf, pooled=True)

    assert base_ci == pytest.approx(stats_ci)


def test_multiple_means_hypothesis():
    sample1 = np.random.normal(50, 1, 10)
    sample2 = np.random.normal(51, 1, 10)
    sample3 = np.random.normal(52, 1, 10)

    workout_result = workout.multiple_mean_hypothesis(sample1, sample2,
                                                      sample3)
    actual_result = means.multiple_mean_hypothesis(sample1, sample2, sample3)

    assert workout_result == pytest.approx(actual_result)
