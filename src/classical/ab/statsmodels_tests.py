
"""AB test functions using statsmodels

These functions are created from using statsmodels library

The module contains useful functions for evaluating AB tests
1. two_sided_props_hypothesis:
    hypothesis test for comparing two proportions
2. two_sided_means_hypothesis:
    hypothesis test for comparing two means
3. two_means_conf_interval:
    confidence interval for difference of two means
4. two_props_conf_interval:
    confidence interval for difference of two proportions

"""


import statsmodels.stats.api as sms
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.stats.proportion as prop_stats


import numpy as np


def one_mean_conf_interval(values: np.ndarray,
                           conf_level: float = 0.95) -> tuple:
    """calculates confidence interval for mean

    Args:
        values (np.array): list of sample values
        conf_level (float, optional): confidence level. Defaults to 0.95.

    Returns:
        tuple: lower and upper bounds of confidence interval
    """
    return sms.DescrStatsW(values).tconfint_mean(alpha=1-conf_level)


def two_sided_means_hypothesis(values1: np.ndarray, values2: np.ndarray,
                               pooled: bool = False) -> tuple:
    """Perform two sided t test to comparing two means

    Args:
        values1 (np.array): sample 1 values
        values2 (np.array): sample 2 values
        pooled (bool, optional): whether to calculate pooled std.
                                 Defaults to False.
    Returns:
        tuple: t_statistic,  p_value of the test
    """
    usevar = 'pooled' if pooled else "unequal"
    (t_statistic, pval, df) = ttest_ind(values1, values2, usevar=usevar)
    return (t_statistic, pval)


def two_means_conf_interval(values1: np.ndarray, values2: np.ndarray,
                            conf_level: float, pooled: bool = False) -> tuple:
    """Calculates confidence interval for the difference of two means

    Args:
        values1 (np.array): sample 1 values
        values2 (np.array): sample 2 values
        conf_level (float): confidence level
        pooled (bool, optional): whether to calculate pooled std.
                                 Defaults to False.

    Returns:
        tuple: lower and upper values of confidence interval
    """
    cm = sms.CompareMeans(sms.DescrStatsW(values1), sms.DescrStatsW(values2))
    alpha = 1 - conf_level
    diff_ci = cm.tconfint_diff(usevar='pooled' if pooled else "unequal",
                               alpha=alpha)
    return diff_ci


def one_prop_conf_interval(values: np.ndarray,
                           conf_level: float = 0.95) -> tuple:
    """calculates confidence interval for a proportion

    Args:
        values (np.array): sample values (values are of values 0 and 1)
        conf_level (float): confidence level

    Returns:
        tuple: lower and upper bounds of confidence interval
    """
    ci = prop_stats.proportion_confint(values.sum(), len(values),
                                       alpha=1-conf_level)
    return ci


def two_sided_props_hypothesis(values1: np.ndarray,
                               values2: np.ndarray) -> tuple:
    """Perform two sided z test to comparing two proportions

    Args:
        values1 (np.array): sample 1 binary(0/1) values
        values2 (np.array): sample 2 binaray(0/1) values

    Returns:
        tuple: z_statistic and p value of the test
    """
    samples = np.vstack((values1, values2))
    counts = samples.sum(axis=1)
    nobs = [np.size(samples, 1)]*2
    return prop_stats.proportions_ztest(counts, nobs)


def two_props_conf_interval(values1: np.ndarray, values2: np.ndarray,
                            conf_level: float) -> tuple:
    """Calculates the confidence interval for the diff between two proportions

    Args:
        values1 (np.array): sample 1 binary(0/1) values
        values2 (np.array): sample 2 binary(0/1) values
        conf_level (float): confidence level

    Returns:
        tuple: lower and upper values of confidence interval
    """
    count1, count2 = values1.sum(), values2.sum()
    nobs1, nobs2 = len(values1), len(values2)
    ci = prop_stats.confint_proportions_2indep(count1, nobs1, count2, nobs2,
                                               alpha=1-conf_level,
                                               method="wald")
    return ci
