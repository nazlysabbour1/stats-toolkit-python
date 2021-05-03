""" Testing AB test functions for proportions

    testing whether base python functions are equivalent to statsmodels ones

"""
import numpy as np
import classical.ab.base_tests as base
import classical.ab.statsmodels_tests as stats

np.random.seed(0)


def test_prop_conf_interval():
    sample1 = np.random.choice(2, 10, p=[0.5, 0.5])
    manual_ci = base.one_prop_conf_interval(sample1)
    stats_ci = stats.one_prop_conf_interval(sample1)
    assert manual_ci == stats_ci


def test_props_hypothesis():
    sample1 = np.random.choice(2, 10, p=[0.5, 0.5])
    sample2 = np.random.choice(2, 10, p=[0.55, 0.45])
    base_result = base.two_sided_props_hypothesis(sample1, sample2)
    stats_result = stats.two_sided_props_hypothesis(sample1, sample2)

    assert base_result == stats_result


def test_props_conf_interval():
    sample1 = np.random.choice(2, 10, p=[0.5, 0.5])
    sample2 = np.random.choice(2, 10, p=[0.55, 0.45])
    conf = 0.95
    base_ci = base.two_props_conf_interval(sample1, sample2,
                                           conf_level=conf)
    stats_ci = stats.two_props_conf_interval(sample1, sample2,
                                             conf_level=conf)

    assert base_ci == stats_ci
