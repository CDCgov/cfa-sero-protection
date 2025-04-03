# Scratchwork

# %% Dependencies
from typing import List

import numpy as np


# %% Function to make titer distribution
def simulate_titers(
    mns: List[float,], sds: List[float,], N: List[float,]
) -> np.ndarray:
    """
    Simulate antibody titers drawn from a (mixture of) lognormal distribution(s).
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


# %% Make a titer distribution

titer_samples = simulate_titers(
    mns=[0.5, 1.0, 1.5, 2.0],
    sds=[0.25, 0.25, 0.25, 0.25],
    N=[100, 100, 100, 100],
)
