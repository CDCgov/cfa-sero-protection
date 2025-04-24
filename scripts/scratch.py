# %% Dependencies
from typing import List

import altair as alt
import numpy as np
import polars as pl


# %% Create functions
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


def calculate_protection(risk_x: pl.Expr, risk_0: pl.Expr):
    """
    Calculate protection using the odds ratio definition,
    using titer = 0 as the baseline for comparison.
    """
    return 1 - (risk_x / (1 - risk_x) / (risk_0 / (1 - risk_0)))


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
    mns=[0.5, 1.0, 1.5, 2.0, 1.5],
    sds=[0.25, 0.25, 0.25, 0.25, 0.5],
    N=[100, 100, 100, 100, 100],
    seed=1,
)

risk_samples = simulate_risk_parameters(
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

# %% Plot titer distribution
titer_dist = (
    alt.Chart(titer_samples[["titer"]])
    .transform_density(
        "titer",
        as_=["Titer", "Density"],
    )
    .mark_line(color="green")
    .encode(
        x="Titer:Q",
        y="Density:Q",
    )
)

titer_dist.display()

# %% Plot risk and protection curves
curves = (
    risk_samples.join(
        pl.DataFrame(
            {
                "dummy_titer": np.arange(
                    np.ceil(titer_samples[["titer"]].max())[0, 0]
                )
            }
        ),
        how="cross",
    )
    .with_columns(
        risk=calculate_risk(
            pl.col("midpoint"),
            pl.col("steepness"),
            pl.col("min_risk"),
            pl.col("max_risk"),
            pl.col("dummy_titer"),
        )
    )
    .with_columns(
        prot=calculate_protection(pl.col("risk"), pl.col("risk").max()).over(
            "id"
        )
    )
)

mean_risk = (
    curves.group_by("dummy_titer")
    .agg(risk=pl.col("risk").mean())
    .sort("dummy_titer")
)

mean_prot = (
    curves.group_by("dummy_titer")
    .agg(prot=pl.col("prot").mean())
    .sort("dummy_titer")
)

alt.data_transformers.disable_max_rows()

risk_curve = alt.Chart(curves).mark_line(opacity=0.3).encode(
    x=alt.X("dummy_titer:Q", title="Titer"),
    y=alt.Y("risk:Q", title="Risk"),
) + alt.Chart(mean_risk).mark_line(opacity=1.0).encode(
    x="dummy_titer:Q",
    y="risk:Q",
)

prot_curve = alt.Chart(curves).mark_line(opacity=0.3).encode(
    x=alt.X("dummy_titer:Q", title="Titer"),
    y=alt.Y("prot:Q", title="Protection"),
) + alt.Chart(mean_prot).mark_line(opacity=1.0).encode(
    x="dummy_titer:Q",
    y="prot:Q",
)

risk_curve.display()
prot_curve.display()

# %% Plot population distributions of risk and protection
pop_dists = (
    risk_samples.join(titer_samples.select("titer"), how="cross")
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
        prot=calculate_protection(pl.col("risk"), pl.col("risk").max()).over(
            "id"
        )
    )
)

alt.data_transformers.disable_max_rows()

risk_dist = (
    alt.Chart(pop_dists.select(["risk", "id"]))
    .transform_density(
        "risk",
        as_=["Risk", "Density"],
        extent=[0, 1],
        bandwidth=0.1,
        groupby=["id"],
    )
    .mark_line(color="green", opacity=0.3)
    .encode(
        x="Risk:Q",
        y="Density:Q",
    )
)

prot_dist = (
    alt.Chart(pop_dists.select(["prot", "id"]))
    .transform_density(
        "prot",
        as_=["Protection", "Density"],
        extent=[0, 1],
        bandwidth=0.1,
        groupby=["id"],
    )
    .mark_line(color="green", opacity=0.3)
    .encode(
        x="Protection:Q",
        y="Density:Q",
    )
)

risk_dist.display()
prot_dist.display()
