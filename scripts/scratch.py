# %% Dependencies
from typing import List

import altair as alt
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
    max_length = max(len(mns), len(sds), len(N))
    mns_arr = np.resize(mns, max_length)
    sds_arr = np.resize(sds, max_length)
    N_arr = np.resize(N, max_length)
    rng = np.random.default_rng(seed)
    titer_samples = np.array([])
    for i in range(len(mns_arr)):
        titer_samples = np.concatenate(
            (
                titer_samples,
                10 ** rng.normal(mns_arr[i], sds_arr[i], N_arr[i]),
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
    for a double-scaled logit risk curve. This posterior is not meant
    to be realistic - it is just an object to work with.
    """

    rng = np.random.default_rng(seed)
    pars = pl.DataFrame(
        {
            "midpoint": rng.normal(midpoint_mn, midpoint_sd, N),
            "steepness": rng.normal(steepness_mn, steepness_sd, N),
            "min_risk": rng.normal(min_risk_mn, min_risk_sd, N),
            "max_risk": rng.normal(max_risk_mn, max_risk_sd, N),
            "id": np.arange(N),
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
    midpoint_mn=80,
    midpoint_sd=10,
    steepness_mn=0.1,
    steepness_sd=0.02,
    min_risk_mn=0.2,
    min_risk_sd=0.04,
    max_risk_mn=0.8,
    max_risk_sd=0.04,
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

# %% Plot titer distribution
alt.Chart(titer_samples[["titer"]]).transform_density(
    "titer",
    as_=["Titer", "Density"],
).mark_line(color="green").encode(
    x="Titer:Q",
    y="Density:Q",
)

# %% Plot risk function
x = pl.DataFrame(
    {"x": np.arange(np.ceil(titer_samples[["titer"]].max())[0, 0])}
)
risk_curves = risk_samples.join(x, how="cross").with_columns(
    y=calculate_risk(
        pl.col("midpoint"),
        pl.col("steepness"),
        pl.col("min_risk"),
        pl.col("max_risk"),
        pl.col("x"),
    )
)
mean_risk_curve = risk_curves.group_by("x").agg(y=pl.col("y").mean()).sort("x")
alt.data_transformers.disable_max_rows()
alt.Chart(risk_curves).mark_line().encode(
    x=alt.X("x", title="Titer"),
    y=alt.Y("y", title="Risk"),
    opacity=alt.value(0.3),
) + alt.Chart(mean_risk_curve).mark_line().encode(
    x="x",
    y="y",
)

# %% Plot protection function
protection_curves = risk_curves.with_columns(
    prot=calculate_protection(pl.col("y"), pl.col("y").max()).over("id")
)
mean_protection_curve = (
    protection_curves.group_by("x").agg(prot=pl.col("prot").mean()).sort("x")
)
alt.Chart(protection_curves).mark_line().encode(
    x=alt.X("x", title="Titer"),
    y=alt.Y("prot", title="Protection"),
    opacity=alt.value(0.3),
) + alt.Chart(mean_protection_curve).mark_line().encode(
    x="x",
    y="prot",
)
