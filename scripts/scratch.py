# %% Dependencies
from typing import List

import numpy as np
import polars as pl


# %% Function to simulate a titer distribution
def simulate_titers(
    mns: List[float,], sds: List[float,], N: List[float,], seed: int
) -> pl.DataFrame:
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

    titers = pl.DataFrame(
        {"titer": titer_samples, "id": np.arange(titer_samples.shape[0]) + 1}
    )

    return titers


# %% Function to calculate risk from parameters and titers
def calculate_risk(
    midpoint: pl.Expr,
    steepness: pl.Expr,
    min_risk: pl.Expr,
    max_risk: pl.Expr,
    titer: pl.Expr,
):
    """
    Calculate risk from the 4 parameters of the double-scaled logit
    and an antibody titer.
    """
    return min_risk + (max_risk - min_risk) / (
        1 + (steepness * (titer - midpoint)).exp()
    )


# %% Function to calculate protection from risk
def calculate_protection(risk_x: pl.Expr, risk_0: pl.Expr):
    """
    Calculate protection using the odds ratio definition,
    using titer = 0 as the baseline for comparison.
    """
    return 1 - (risk_x / (1 - risk_x) / (risk_0 / (1 - risk_0)))


# %% Function to simulate parameters of a risk curve
def simulate_risk_parameters(
    midpoint_hi: float,
    midpoint_lo: float,
    steepness_hi: float,
    steepness_lo: float,
    min_risk_hi: float,
    min_risk_lo: float,
    max_risk_hi: float,
    max_risk_lo: float,
    N: int,
    seed: int,
):
    """
    Simulate N draws from a posterior distribution of fit parameters
    for a double-scaled logit risk curve. This posterior is not meant
    to be realistic - it is just an object to work with.
    """

    rng = np.random.default_rng(seed)
    pars = pl.DataFrame(
        {
            "midpoint": rng.uniform(midpoint_lo, midpoint_hi, N),
            "steepness": rng.uniform(steepness_lo, steepness_hi, N),
            "min_risk": rng.uniform(min_risk_lo, min_risk_hi, N),
            "max_risk": rng.uniform(max_risk_lo, max_risk_hi, N),
        }
    )

    return pars


# %% Simulate a titer distribution and a risk curve
titer_samples = simulate_titers(
    mns=[0.5, 1.0, 1.5, 2.0],
    sds=[0.25, 0.25, 0.25, 0.25],
    N=[100, 100, 100, 100],
    seed=1,
)

risk_samples = simulate_risk_parameters(
    midpoint_hi=100,
    midpoint_lo=75,
    steepness_hi=3,
    steepness_lo=1,
    min_risk_hi=0.25,
    min_risk_lo=0.15,
    max_risk_hi=0.90,
    max_risk_lo=0.80,
    N=100,
    seed=1,
)

# %% Calculate risk and protection for each posterior sample for each titer
protection_samples = (
    risk_samples.join(titer_samples, how="cross")
    .with_columns(
        risk=calculate_risk(
            pl.col("midpoint"),
            pl.col("steepness"),
            pl.col("min_risk"),
            pl.col("max_risk"),
            pl.col("titer"),
        )
    )
    .with_columns(
        protection=calculate_protection(pl.col("risk"), pl.col("max_risk"))
    )
)
