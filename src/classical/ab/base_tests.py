"""AB test functions using base python

These functions are created from using basic python libraries

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
import math
import numpy as np
import scipy.stats


def get_z_critical(conf_level: float) -> float:
    """Gets z critical , which is z value corresponding to a specific
    confidence level (for a two sided test)

    Args:
        conf_level (float): confidence level

    Returns:
        float: z critical value
    """
    return scipy.stats.norm.ppf(1-(1-conf_level)/2)


def get_t_critical(conf_level: float, df: int) -> float:
    """Gets t critical , which is t value corresponding to a specific
    confidence level (for a two sided test)

    Args:
        conf_level (float): confidence level
        df (int): degrees of freedom

    Returns:
        float: t critical value
    """
    return scipy.stats.t.ppf(1-(1-conf_level)/2, df)


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
    t_critical = get_t_critical(conf_level, n-1)
    print(t_critical, x_bar, se, sd)
    lower_ci = x_bar - t_critical * se
    upper_ci = x_bar + t_critical * se
    return (lower_ci, upper_ci)


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


def two_sided_means_hypothesis(values1: np.ndarray, values2: np.ndarray,
                               pooled: bool = False) -> tuple:
    """Perform two sided t test to comparing two means

    Args:
        values1 (np.array): sample 1 values
        values2 (np.array): sample 2 values
        pooled (bool, optional): whether to calculate pooled std.
                                 Defaults to False.
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
    p_value = scipy.stats.t.sf(abs(t_statistic), df=df)*2
    return (t_statistic, p_value)


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
    x_bar1 = np.mean(values1)
    x_bar2 = np.mean(values2)
    x_diff = x_bar1 - x_bar2
    n1, n2 = len(values1), len(values2)
    se = __get_two_sample_standard_error(values1, values2, pooled)
    df = (n1 + n2 - 2) if pooled else min(n1 - 1, n2 - 1)

    t_critical = get_t_critical(conf_level, df)
    lower_ci = x_diff - t_critical * se
    upper_ci = x_diff + t_critical * se
    return (lower_ci, upper_ci)


def one_prop_conf_interval(values: np.ndarray,
                           conf_level: float = 0.95) -> tuple:
    """calculates confidence interval for a proportion

    Args:
        values (np.array): sample values (values are of values 0 and 1)
        conf_level (float): confidence level

    Returns:
        tuple: lower and upper values of confidence interval
    """
    p = np.mean(values)
    n = len(values)
    se = math.sqrt(p*(1-p)/n)
    z_critical = get_z_critical(conf_level)
    lower_ci = p - z_critical * se
    upper_ci = p + z_critical * se
    return (lower_ci, upper_ci)


def two_sided_props_hypothesis(values1: np.ndarray,
                               values2: np.ndarray) -> tuple:
    """Perform two sided z test to comparing two proportions

    Args:
        values1 (np.array): sample 1 binary(0/1) values
        values2 (np.array): sample 2 binaray(0/1) values

    Returns:
        tuple: z_statistic and p value of the test
    """
    p1 = np.mean(values1)
    p2 = np.mean(values2)
    n1 = len(values1)
    n2 = len(values2)
    p_pool = (np.sum(values1) + np.sum(values2))/(n1+n2)
    se = math.sqrt((p_pool*(1-p_pool))/n1 + (p_pool*(1-p_pool))/n2)
    z_statistic = (p1-p2)/se
    p_value = (1 - scipy.stats.norm.cdf(abs(z_statistic)))*2
    return (z_statistic, p_value)


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
    p1 = np.mean(values1)
    p2 = np.mean(values2)
    p_diff = p1-p2
    n1 = len(values1)
    n2 = len(values2)
    SE = math.sqrt(((p1*(1-p1))/n1) + ((p2*(1-p2))/n2))
    z_critical = get_z_critical(conf_level)
    lower = p_diff - z_critical*SE
    upper = p_diff + z_critical*SE
    return (lower, upper)
