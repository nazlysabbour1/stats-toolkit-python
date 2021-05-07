"""Workout for Statistical tests to estimate population proportions

These functions are created from using basic python libraries

The module contains useful functions for evaluating AB tests
1. one_prop_conf_interval
2. one_prop_hypothesis
3. two_props_diff_conf_interval
4. two_props_hypothesis

"""
import math
import numpy as np
import classical.workout.utils as utils


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
    z_critical = utils.get_z_critical(conf_level)
    lower_ci = p - z_critical * se
    upper_ci = p + z_critical * se
    return (lower_ci, upper_ci)


def one_prop_hypothesis(values: np.ndarray,
                        null_val: float = 0,
                        alternative: str = "two-sided"):
    p = np.mean(values)
    n = len(values)
    se = math.sqrt(null_val*(1-null_val)/n)
    z_statistic = (p - null_val)/se
    p_value = utils.get_norm_pvalue(z_statistic, alternative)
    return (z_statistic, p_value)


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
    p1 = np.mean(values1)
    p2 = np.mean(values2)
    p_diff = p1-p2
    n1 = len(values1)
    n2 = len(values2)
    SE = math.sqrt(((p1*(1-p1))/n1) + ((p2*(1-p2))/n2))
    z_critical = utils.get_z_critical(conf_level)
    lower = p_diff - z_critical*SE
    upper = p_diff + z_critical*SE
    return (lower, upper)


def two_props_hypothesis(values1: np.ndarray,
                         values2: np.ndarray,
                         alternative: str = "two-sided") -> tuple:
    """Perform two sided z test to comparing two proportions

    Args:
        values1 (np.array): sample 1 binary(0/1) values
        values2 (np.array): sample 2 binaray(0/1) values
        alternative (str, optional): two-sided/larger/smaller.
                                     Defaults to "two-sided".

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
    p_value = utils.get_norm_pvalue(z_statistic, alternative)
    return (z_statistic, p_value)
