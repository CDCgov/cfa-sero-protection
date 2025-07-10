import numpy as np


def calculate_risk(
    titer: float,
    slope: float,
    midpoint: float,
    min_risk: float,
    max_risk: float,
) -> float:
    """
    Define relative risk as a function of log antibody titer,
    following a double-scaled logit functional form.

    Eventually this should operate on lists or polars columns,
    not individual float numbers.
    """

    return min_risk + (max_risk - min_risk) / (
        1 + np.exp(slope * (titer - midpoint))
    )
