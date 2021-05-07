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


def get_t_pvalue(t_value: float,
                 df: int,
                 alternative: str = "two-sided") -> float:
    """Get corresponding p value for t value depending on test type

    Args:
        t_value (float): t value
        df (int): degrees of freedom
        alternative (str, optional): two-sided/larger/smaller.
                                     Defaults to "two-sided".

    Raises:
        ValueError: when the test type is invalid

    Returns:
        float: p value
    """

    if alternative == "two-sided":
        p_value = scipy.stats.t.sf(abs(t_value), df=df)*2
    elif alternative == "larger":
        p_value = scipy.stats.t.sf(t_value, df=df)
    elif alternative == "smaller":
        p_value = scipy.stats.t.cdf(t_value, df=df)
    else:
        raise ValueError("invalid alternative")
    return p_value


def get_norm_pvalue(z_value: float,
                    alternative: str = "two-sided") -> float:
    """Get corresponding p value for z value depending on test type

    Args:
        z_value (float): z value
        alternative (str, optional): two-sided/larger/smaller.
                                     Defaults to "two-sided".

    Raises:
        ValueError: when the test type is invalid

    Returns:
        float: p value
    """

    if alternative == "two-sided":
        p_value = scipy.stats.norm.sf(abs(z_value))*2
    elif alternative == "larger":
        p_value = scipy.stats.norm.sf(z_value)
    elif alternative == "smaller":
        p_value = scipy.stats.norm.cdf(z_value)
    else:
        raise ValueError("invalid alternative")
    return p_value


def get_f_pvalue(f_value: float, dfg: int, dfe: int):
    return scipy.stats.f.sf(f_value, dfg, dfe)
