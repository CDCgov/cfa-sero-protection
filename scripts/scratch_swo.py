# %% Dependencies
from typing import List

import altair as alt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde


# %% Create functions
# so long as these are the same length, rng.normal() can be called
# using them in parallel
def simulate_titers(
    mns: np.ndarray, sds: np.ndarray, n_per_dist: int, seed: int
) -> np.ndarray:
    """
    Simulate antibody titers drawn from a (mixture of) lognormal distribution(s).
    This is just to have some numbers to use, not necessarily to be realistic.
    """
    n_dist = len(mns)  # number of distributions
    assert n_dist == len(sds)
    rng = np.random.default_rng(seed)
    titer_samples = 10 ** rng.normal(
        loc=mns, scale=sds, size=(n_per_dist, n_dist)
    ).reshape(n_per_dist * n_dist)

    return titer_samples


# I think it's easier to think of this as a fixed set of parameters
# and an array of titers. Also, using np.exp(), rather than .exp(),
# means this can work with numpy arrays and pl Exprs (although pylance
# will complain)
def calculate_risk(
    titer: np.ndarray,
    midpoint: float,
    steepness: float,
    min_risk: float,
    max_risk: float,
) -> np.ndarray:
    """
    Calculate risk from the 4 parameters of the double-scaled logit
    and an antibody titer.
    """
    return min_risk + (max_risk - min_risk) / (
        1 + np.exp(steepness * (titer - midpoint))
    )


# ditto here, we can make a single call to rng.normal().
# I think it's actually easier to keep this as a Python object,
# because the generation of the risk distributions, which needs
# to go through the KDE, is easier to do this way, than in a
# polars object. (If you *did* want to do it in polars, I would
# have this produce a Series of structs, and then you would
# use map_elements() to get the distributions as structs, then
# unpack those... which is complicated.)
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
) -> List[dict]:
    """
    Simulate draws from a posterior distribution of fit parameters
    for a double-scaled logit risk curve. This posterior is not meant
    to be realistic - it is just an object to work with.
    """

    rng = np.random.default_rng(seed)
    samples = rng.normal(
        loc=np.array([midpoint_mn, steepness_mn, min_risk_mn, max_risk_mn]),
        scale=np.array([midpoint_sd, steepness_sd, min_risk_sd, max_risk_sd]),
        size=(N, 4),
    )

    return [
        {
            "midpoint": s[0],
            "steepness": s[1],
            "min_risk": s[2],
            "max_risk": s[3],
        }
        for s in samples
    ]


# %% Simulate a titer distribution and plot it
titer_samples = simulate_titers(
    mns=np.array([0.5, 1.0, 1.5, 2.0, 1.5]),
    sds=np.array([0.25, 0.25, 0.25, 0.25, 0.5]),
    n_per_dist=100,
    seed=1,
)

# maybe just do a histogram? Altair defaults to too few bins
# with just bin=True, so I needed to specify the step size
titer_dist = (
    alt.Chart(pl.DataFrame({"titer": titer_samples}))
    .mark_bar()
    .encode(alt.X("titer", bin=alt.Bin(step=10)), alt.Y("count()"))
)

# I didn't do the protection curves, just for simplicity.
# although I did wonder why you want .over(titer_id) in there?
# This means that "protection" depends on the maximum risk for
# that parameterization, which I found confusing

titer_dist.display()

# %% Simulate a risk curve and plot it
parameter_samples = simulate_risk_parameters(
    midpoint_mn=120,
    midpoint_sd=10,
    steepness_mn=0.05,
    steepness_sd=0.01,
    min_risk_mn=0.2,
    min_risk_sd=0.04,
    max_risk_mn=0.8,
    max_risk_sd=0.04,
    N=100,
    seed=1,
)

curves = (
    pl.from_dicts(parameter_samples)
    .with_row_index("parameter_id")
    .join(
        pl.DataFrame({"titer": np.linspace(0, titer_samples.max(), 100)}),
        how="cross",
    )
    .with_columns(
        # this is where pylance complains, because it doesn't know that
        # I can also put Exprs into this function
        risk=calculate_risk(
            pl.col("titer"),
            pl.col("midpoint"),
            pl.col("steepness"),
            pl.col("min_risk"),
            pl.col("max_risk"),
        )
    )
)

# polars defaults to using the same name for columns, so `pl.col('risk').mean()`
# will by default have name "risk"
mean_risk = curves.group_by("titer").agg(pl.col("risk").mean())

alt.data_transformers.disable_max_rows()

risk_curve = alt.Chart(curves).mark_line(opacity=0.1).encode(
    # altair now recommends that you use the alt.Whatever()
    # syntax, rather than the old keywords. I've personally
    # found this clearer in the end.
    alt.X("titer:Q", title="Titer"),
    alt.Y("risk:Q", title="Risk"),
    alt.Detail("parameter_id"),
) + alt.Chart(mean_risk).mark_line(opacity=1.0, color="black").encode(
    alt.X("titer:Q"),
    alt.Y("risk:Q"),
)

risk_curve.display()

# %% Calculate and plot population distributions of risk and protection


# for each parameter sample, calculate the risks for each sampled ("in") titers,
# then use the KDE to approximate the density at the "out" titers
def risk_density(params, out_titers, in_titers=titer_samples) -> pl.DataFrame:
    in_risks = calculate_risk(in_titers, **params)
    empirical_density = gaussian_kde(in_risks)
    out_risks = empirical_density(out_titers)

    return pl.DataFrame({"titer": out_titers, "risk_dens": out_risks})


# this is the bit that is a *lot* easier if we keep the parameters as a list
# of dicts, rather than as columns in polars
densities = pl.concat(
    [
        risk_density(params, np.linspace(0, 1, 1000)).with_columns(
            parameter_id=i
        )
        for i, params in enumerate(parameter_samples)
    ]
)

mean_pop_dens = densities.group_by("titer").agg(pl.col("risk_dens").mean())

risk_dist = alt.Chart(densities).mark_line(color="green", opacity=0.1).encode(
    x="titer:Q", y="risk_dens:Q", detail="parameter_id"
) + alt.Chart(mean_pop_dens).mark_line(color="black").encode(
    x=alt.X("titer:Q", title="Risk"), y=alt.Y("risk_dens:Q", title="Density")
)

risk_dist.display()
