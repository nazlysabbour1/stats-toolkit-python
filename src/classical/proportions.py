"""Statistical tests to estimate population proportions using statsmodels

These functions are created from using basic python libraries

The module contains useful functions for evaluating AB tests
1. one_prop_conf_interval
2. one_prop_hypothesis
3. two_props_diff_conf_interval
4. two_props_hypothesis

"""

import numpy as np
import statsmodels.stats.proportion as prop_stats


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


def one_prop_hypothesis(values: np.ndarray,
                        null_val: float = 0,
                        alternative: str = "two-sided") -> tuple:
    """[z test for comparing the proportion of 1 population with null value

    Args:
        values (np.ndarray): sample values
        null_val (float, optional): null value. Defaults to 0.
        alternative (str, optional): two-sided/larger/smaller.
                                     Defaults to "two-sided".

    Returns:
        tuple: z statistic, p value
    """
    count = np.sum(values)
    nob = len(values)
    return prop_stats.proportions_ztest(count, nob, alternative=alternative)


def two_props_diff_conf_interval(values1: np.ndarray, values2: np.ndarray,
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


def two_props_hypothesis(values1: np.ndarray,
                         values2: np.ndarray,
                         alternative: str = "two-sided") -> tuple:
    """z test for comparing two proportions

    Args:
        values1 (np.array): sample 1 binary(0/1) values
        values2 (np.array): sample 2 binaray(0/1) values
        alternative (str, optional): two-sided/larger/smaller.
                                     Defaults to "two-sided".

    Returns:
        tuple: z_statistic and p value of the test
    """
    samples = np.vstack((values1, values2))
    counts = samples.sum(axis=1)
    nobs = [np.size(samples, 1)]*2
    return prop_stats.proportions_ztest(counts, nobs, alternative=alternative)
