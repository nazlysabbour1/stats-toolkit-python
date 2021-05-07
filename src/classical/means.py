"""Statistical tests to estimate population means using statsmodels

These functions are created from using basic python libraries

The module contains useful functions for evaluating AB tests
1. one_mean_conf_interval
2. one_mean_hypothesis
3. two_means_diff_conf_interval
4. two_means_hypothesis

"""
import numpy as np
import statsmodels.stats.api as sms
from statsmodels.stats.weightstats import ttest_ind


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


def one_mean_hypothesis(values: np.ndarray, null_val: float = 0,
                        alternative: str = "two-sided") -> tuple:
    """Null hypothesis testing that mean of population is equal to null_val

    Args:
        values (np.ndarray): sample values
        null_val (float, optional): null value. Defaults to 0.
        alternative (str, optional): two-sided/larger/smaller.
                                     Defaults to "two-sided".

    Returns:
        tuple: t statistic,  p_value of the test
    """
    stats = sms.DescrStatsW(values)
    tstat, pvalue, _ = stats.ttest_mean(null_val, alternative=alternative)
    return (tstat, pvalue)


def two_means_diff_conf_interval(values1: np.ndarray, values2: np.ndarray,
                                 conf_level: float,
                                 pooled: bool = False) -> tuple:
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


def two_means_hypothesis(values1: np.ndarray, values2: np.ndarray,
                         pooled: bool = False,
                         alternative: str = "two-sided") -> tuple:
    """Perform t test  comparing two means

    Args:
        values1 (np.array): sample 1 values
        values2 (np.array): sample 2 values
        pooled (bool, optional): whether to calculate pooled std.
                                 Defaults to False.
        alternative (str, optional): two-sided/larger/smaller.
                                     Defaults to "two-sided".
    Returns:
        tuple: t statistic,  p_value of the test
    """
    usevar = 'pooled' if pooled else "unequal"
    (tstat, pval, df) = ttest_ind(values1, values2, usevar=usevar,
                                  alternative=alternative)
    return (tstat, pval)
