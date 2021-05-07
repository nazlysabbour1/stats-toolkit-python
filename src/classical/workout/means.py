"""Workout of Statistical tests to estimate population means

These functions are created from using basic python libraries

The module contains useful functions for evaluating AB tests
1. one_mean_conf_interval
2. one_mean_hypothesis
3. two_means_diff_conf_interval
4. two_means_hypothesis

"""

import math
import numpy as np
import classical.workout.utils as utils


def one_mean_conf_interval(values: np.ndarray,
                           conf_level: float = 0.95) -> tuple:
    """calculates confidence interval for mean

    Args:
        values (np.array): list of sample values
        conf_level (float, optional): confidence level. Defaults to 0.95.

    Returns:
        tuple: lower and upper bounds of confidence interval
    """
    x_bar = np.mean(values)
    sd = np.std(values,  ddof=1)
    n = len(values)
    se = sd/math.sqrt(n)
    t_critical = utils.get_t_critical(conf_level, n-1)
    lower_ci = x_bar - t_critical * se
    upper_ci = x_bar + t_critical * se
    return (lower_ci, upper_ci)


def one_mean_hypothesis(values: np.ndarray,
                        null_val: float = 0,
                        alternative: str = "two-sided") -> tuple:
    x_bar = np.mean(values)
    sd = np.std(values, ddof=1)
    n = len(values)
    se = sd/math.sqrt(n)
    t_statistic = (x_bar-null_val)/se
    df = n - 1
    p_value = utils.get_t_pvalue(t_statistic, df, alternative)
    return (t_statistic, p_value)


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
    x_bar1 = np.mean(values1)
    x_bar2 = np.mean(values2)
    x_diff = x_bar1 - x_bar2
    n1, n2 = len(values1), len(values2)
    se = __get_two_sample_standard_error(values1, values2, pooled)
    df = (n1 + n2 - 2) if pooled else min(n1 - 1, n2 - 1)

    t_critical = utils.get_t_critical(conf_level, df)
    lower_ci = x_diff - t_critical * se
    upper_ci = x_diff + t_critical * se
    return (lower_ci, upper_ci)


def two_means_hypothesis(values1: np.ndarray, values2: np.ndarray,
                         pooled: bool = False,
                         alternative: str = "two-sided") -> tuple:
    """Perform two sided t test to comparing two means

    Args:
        values1 (np.array): sample 1 values
        values2 (np.array): sample 2 values
        pooled (bool, optional): whether to calculate pooled std.
                                 Defaults to False.
        alternative (str, optional): two-sided/larger/smaller.
                                     Defaults to "two-sided".
    Returns:
        tuple: t_statistic and p_value of the test
    """
    x_bar1 = np.mean(values1)
    x_bar2 = np.mean(values2)
    x_diff = x_bar1 - x_bar2
    n1, n2 = len(values1), len(values2)

    se = __get_two_sample_standard_error(values1, values2, pooled)
    df = (n1 + n2 - 2) if pooled else min(n1 - 1, n2 - 1)

    t_statistic = x_diff/se

    p_value = utils.get_t_pvalue(t_statistic, df, alternative)

    return (t_statistic, p_value)


def multiple_mean_hypothesis(*args) -> tuple:
    """Computes Anova test to get whether the mean of at least one group is 
    different
    Args:
        *args consecutive sample group values each group sample
              is represented by a list
    Returns:
        tuple: f statistic, p value
    """
    groups = [np.asarray(arg, dtype=float) for arg in args]
    all = np.concatenate(groups)
    n_g = len(groups)
    n_t = len(all)
    y_bar = np.mean(all)
    sst = np.sum((all-y_bar)**2)
    y_barg = list(map(np.mean, groups))
    n_per_g = list(map(len, groups))
    ssg = sum([n_per_g[i] * (y_g-y_bar)**2 for i, y_g in enumerate(y_barg)])

    sse = sst - ssg
    df_t = n_t - 1
    df_g = n_g - 1
    df_e = df_t - df_g
    mse = sse / df_e
    msg = ssg / df_g
    f_statistic = msg/mse
    p_value = utils.get_f_pvalue(f_statistic, df_g, df_e)
    return (f_statistic, p_value)


def __get_two_sample_standard_error(values1: np.ndarray, values2: np.ndarray,
                                    pooled: bool = False) -> float:
    """Calculates standard error for the mean of two samples

    Args:
        values1 (np.array): first sample values
        values2 (np.array): second sample values
        pooled (bool, optional): whether to calculate pooled estimate.
                                 Defaults to False.

    Returns:
        float: standard error
    """
    n1 = len(values1)
    n2 = len(values2)
    s1 = np.std(values1, ddof=1)
    s2 = np.std(values2, ddof=1)
    if pooled:
        s_pool = __get_pooled_standard_deviation(s1, s2, n1, n2)
        s1, s2 = s_pool, s_pool

    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    return se


def __get_pooled_standard_deviation(s1: float, s2: float,
                                    n1: int, n2: int) -> float:
    """Calcultes pooled standard deviation

    Args:
        s1 (float): standard deviation of sample 1
        s2 (float): standard deviation of sample 2
        n1 (int): no observations of sample 1
        n2 (int): no observations of sample 2

    Returns:
        float: pooled standard deviation
    """
    s_pool = math.sqrt((s1**2 * (n1 - 1) + s2**2 * (n2 - 1)) /
                       (n1 + n2 - 2))
    return s_pool
