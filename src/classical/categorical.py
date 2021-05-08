""" Statistical tests for evaluating distribution of 1 or 2 categorical
variables with at least one having more than two levels

functions:

1. one_categorical_hypothesis
2. two_categorical_hypothesis
"""

import numpy as np
import scipy.stats


def one_categorical_hypothesis(counts: np.ndarray, nobs: np.ndarray) -> tuple:
    """Applying chi square test goodness of fit

    Ho: the observed counts of the input groups follow population distribution
    HA: the observed counts of groups do not follow population distribution
        (not random pick form population)

    Args:
        counts (np.ndarray): input group  observed counts
        nobs (np.ndarray): input group total count

    Returns:
        tuple: chi square value, p value
    """
    p_expected = sum(counts) / sum(nobs)
    expected_counts = nobs * p_expected
    result = scipy.stats.chisquare(counts, expected_counts)
    chi_square, p_value = result.statistic, result.pvalue
    return chi_square, p_value


def two_categorical_hypothesis(observed: np.ndarray) -> tuple:
    """Applying chi square independence test to compare two variables

    Ho: two variables are independent
    Ha: two variables are dependent

    Args:
        observed (np.ndarray): 2d array the rows represent first variable
                                        the columns represent second variable

    Returns:
        tuple: chi square value, p value
    """
    chi_square, p_value, _, _ = scipy.stats.chi2_contingency(observed)
    return chi_square, p_value
