# %% Dependencies
from typing import List

import numpy as np


# %% Function to simulate a titer distribution
def simulate_titers(
    mns: List[float,], sds: List[float,], N: List[float,], seed: int
) -> np.ndarray:
    """
    Simulate antibody titers drawn from a (mixture of) lognormal distribution(s).
    This is just to have some numbers to use, not necessarily to be realistic.
    """
    assert len(mns) == len(sds) & len(mns) == len(N), (
        "Means, standard deviations, and sample size must all be the same length."
    )
    rng = np.random.default_rng(seed)
    titer_samples = np.array([])
    for i in range(len(mns)):
        titer_samples = np.concatenate(
            (
                titer_samples,
                10 ** rng.normal(mns[i], sds[i], N[i]),
            ),
            axis=0,
        )

    return titer_samples


# %% Function to calculate risk from parameters and titers
def calculate_risk(
    midpoint: float,
    steepness: float,
    min_risk: float,
    max_risk: float,
    titer: float,
):
    """
    Calculate risk from the 4 parameters of the double-scaled logit
    and an antibody titer.
    """
    return min_risk + (max_risk - min_risk) / (
        1 + np.exp(steepness * (titer - midpoint))
    )


# %% Function to calculate protection from risk
def calculate_protection(risk_x: float, risk_0: float):
    """
    Calculate protection using the odds ratio definition,
    using titer = 0 as the baseline for comparison.
    """
    return 1 - (risk_x / (1 - risk_x) / (risk_0 / (1 - risk_0)))


# %% Function to simulate parameters of a risk curve
def simulate_risk_parameters(
    midpoint_mn: float,
    midpoint_sd: float,
    steepness_mn: float,
    steepness_sd: float,
    min_risk_mn: float,
    min_risk_sd: float,
    max_risk_mn: float,
    max_risk_sd: float,
    N: int,
    seed: int,
):
    """
    Simulate N draws from a posterior distribution of fit parameters
    for a double-scaled logit risk curve
    """

    return None


# %% Make a titer distribution
titer_samples = simulate_titers(
    mns=[0.5, 1.0, 1.5, 2.0],
    sds=[0.25, 0.25, 0.25, 0.25],
    N=[100, 100, 100, 100],
    seed=1,
)
