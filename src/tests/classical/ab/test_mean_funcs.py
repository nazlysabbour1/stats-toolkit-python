""" Testing AB test functions for means

    testing whether base python functions are equivalent to statsmodels ones

"""
import numpy as np
import classical.ab.base_tests as base
import classical.ab.statsmodels_tests as stats
import testing_utils as u
np.random.seed(0)


def test_mean_conf_interval():
    sample1 = np.random.normal(50, 1, 10)
    manual_ci = base.one_mean_conf_interval(sample1)
    stats_ci = stats.one_mean_conf_interval(sample1)
    assert manual_ci == stats_ci


def test_means_hypothesis():
    sample1 = np.random.normal(50, 1, 10)
    sample2 = np.random.normal(51, 1, 10)
    base_result = base.two_sided_means_hypothesis(sample1, sample2,
                                                  pooled=True)
    stats_result = stats.two_sided_means_hypothesis(sample1, sample2,
                                                    pooled=True)

    assert u.apply_round(base_result) == u.apply_round(stats_result)


def test_means_conf_interval():
    sample1 = np.random.normal(50, 1, 10)
    sample2 = np.random.normal(51, 1, 10)
    conf = 0.95
    base_ci = base.two_means_conf_interval(sample1, sample2,
                                           conf_level=conf, pooled=True)
    stats_ci = stats.two_means_conf_interval(sample1, sample2,
                                             conf_level=conf, pooled=True)

    assert base_ci == stats_ci
