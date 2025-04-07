# %% Dependencies
from typing import List

import numpy as np


# %% Function to simulate a titer distribution
def simulate_titers(
    mns: List[float,], sds: List[float,], N: List[float,]
) -> np.ndarray:
    """
    Simulate antibody titers drawn from a (mixture of) lognormal distribution(s).
    This is just to have some numbers to use, not necessarily to be realistic.
    """
    assert len(mns) == len(sds) & len(mns) == len(N), (
        "Means, standard deviations, and sample size must all be the same length."
    )
    titer_samples = np.array([])
    for i in range(len(mns)):
        titer_samples = np.concatenate(
            (
                titer_samples,
                10 ** np.random.normal(mns[i], sds[i], N[i]),
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
    return min_risk + (max_risk - min_risk) / (
        1 + np.exp(steepness * (titer - midpoint))
    )


# %% Function to simulate a protection curve


# %% Make a titer distribution
titer_samples = simulate_titers(
    mns=[0.5, 1.0, 1.5, 2.0],
    sds=[0.25, 0.25, 0.25, 0.25],
    N=[100, 100, 100, 100],
)
