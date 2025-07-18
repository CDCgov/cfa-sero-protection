# %% Import modules
import sys

import altair as alt
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample

sys.path.append("/home/tec0/cfa-sero-protection")
import seropro.samples as sps

# %% Define parameters
POP_SIZE = 10000
NUM_DAYS = 100
AB_RISK_SLOPE = 0.02
AB_RISK_MIDPOINT = 500.0
AB_RISK_MIN = 0.1
AB_RISK_MAX = 0.9
AB_RISK_MIDPOINT_DROP = 2.0
AB_DECAY = [0.9, 1.0]
AB_SPIKE = [500.0, 1000.0]
AB_LAG = 4
TC_RISK_SLOPE = 0.02
TC_RISK_MIDPOINT = 500.0
TC_RISK_MIN = 0.1
TC_RISK_MAX = 0.9
TC_RISK_MIDPOINT_DROP = 2.0
TC_DECAY = [0.9, 1.0]
TC_SPIKE = [500.0, 1000.0]
TC_LAG = 4
FORCE_EXP = 0.1
FORCE_VAX = 0.02
RECOVERY = 0.25
XSEC_SIZE = 100
XSEC_WINDOW = [50, 57]
TND_INF_PRB = 0.1
TND_NON_PRB = 0.01


# %% Define functions
def calculate_risk(
    titer: pl.Expr,
    slope: float,
    midpoint: float,
    min_risk: float,
    max_risk: float,
) -> pl.Expr:
    """
    Define relative risk as a function of log antibody titer,
    following a double-scaled logit functional form.
    """

    return min_risk + (max_risk - min_risk) / (
        1 + (slope * (titer - midpoint)).exp()
    )


# %% Run simulation
daily_data = []

for i in range(NUM_DAYS):
    if i == 0:
        new_daily_data = pl.DataFrame(
            {
                "id": range(0, POP_SIZE),
                "inf_status": [False] * POP_SIZE,
                "vax_status": [False] * POP_SIZE,
                "ab": [0.0] * POP_SIZE,
                "tc": [0.0] * POP_SIZE,
                "inf_new_lag_ab": [False] * POP_SIZE,
                "vax_new_lag_ab": [False] * POP_SIZE,
                "inf_new_lag_tc": [False] * POP_SIZE,
                "vax_new_lag_tc": [False] * POP_SIZE,
            }
        )
    else:
        new_daily_data = (
            daily_data[i - 1]
            .with_columns(
                ab=pl.col("ab")
                * pl.Series(
                    "ab", np.random.uniform(AB_DECAY[0], AB_DECAY[1], POP_SIZE)
                ),
                tc=pl.col("tc")
                * pl.Series(
                    "tc", np.random.uniform(TC_DECAY[0], TC_DECAY[1], POP_SIZE)
                ),
                inf_new_lag_ab=daily_data[max(0, i - AB_LAG)]["inf_new"],
                vax_new_lag_ab=daily_data[max(0, i - AB_LAG)]["vax_new"],
                inf_new_lag_tc=daily_data[max(0, i - TC_LAG)]["inf_new"],
                vax_new_lag_tc=daily_data[max(0, i - TC_LAG)]["vax_new"],
            )
            .with_columns(
                ab=pl.when(pl.col("inf_new_lag_ab") | pl.col("vax_new_lag_ab"))
                .then(
                    pl.Series(
                        "ab",
                        np.random.uniform(AB_SPIKE[0], AB_SPIKE[1], POP_SIZE),
                    )
                )
                .otherwise(pl.col("ab")),
                tc=pl.when(pl.col("inf_new_lag_tc") | pl.col("vax_new_lag_tc"))
                .then(
                    pl.Series(
                        "tc",
                        np.random.uniform(TC_SPIKE[0], TC_SPIKE[1], POP_SIZE),
                    )
                )
                .otherwise(pl.col("tc")),
            )
        )

    new_daily_data = (
        new_daily_data.with_columns(
            ab_risk=calculate_risk(
                pl.col("ab"),
                AB_RISK_SLOPE,
                AB_RISK_MIDPOINT - (AB_RISK_MIDPOINT_DROP * i),
                AB_RISK_MIN,
                AB_RISK_MAX,
            ),
            tc_risk=calculate_risk(
                pl.col("tc"),
                TC_RISK_SLOPE,
                TC_RISK_MIDPOINT - (TC_RISK_MIDPOINT_DROP * i),
                TC_RISK_MIN,
                TC_RISK_MAX,
            ),
            inf_draw=pl.Series("inf_draw", np.random.rand(POP_SIZE))
            / FORCE_EXP,
            vax_draw=pl.Series("vax_draw", np.random.rand(POP_SIZE))
            / FORCE_VAX,
            rec_draw=pl.Series("rec_draw", np.random.rand(POP_SIZE))
            / RECOVERY,
            day=i,
        )
        .with_columns(
            inf_new=pl.when(
                (pl.col("inf_draw") < pl.min_horizontal("ab_risk", "tc_risk"))
                & ~pl.col("inf_status")
            )
            .then(True)
            .otherwise(False)
        )
        .with_columns(
            inf_status=pl.col("inf_status") | pl.col("inf_new"),
        )
        .with_columns(
            vax_new=pl.when(
                (pl.col("vax_draw") < 1.0)
                & ~pl.col("inf_status")
                & ~pl.col("vax_status")
            )
            .then(True)
            .otherwise(False)
        )
        .with_columns(vax_status=pl.col("vax_status") | pl.col("vax_new"))
        .with_columns(
            rec_new=pl.when((pl.col("rec_draw") < 1.0) & ~pl.col("inf_new"))
            .then(True)
            .otherwise(False)
        )
        .with_columns(
            inf_status=pl.when(pl.col("rec_new"))
            .then(False)
            .otherwise(pl.col("inf_status"))
        )
    )

    daily_data.append(new_daily_data)

all_data = pl.concat(daily_data).select(
    "id",
    "day",
    "inf_status",
    "inf_new",
    "vax_status",
    "ab",
    "tc",
)

# %% Plot number of people currently infected through time
inf_plot = all_data.group_by("day").agg(pl.col("inf_status").sum())
alt.Chart(inf_plot).mark_line().encode(
    x=alt.X("day:Q", title="Day"),
    y=alt.Y("inf_status:Q", title="Current Number Infected"),
)

# %% Plot number of people cumulatively vaccinated through time
vax_plot = all_data.group_by("day").agg(pl.col("vax_status").sum())
alt.Chart(vax_plot).mark_line().encode(
    x=alt.X("day:Q", title="Day"),
    y=alt.Y("vax_status:Q", title="Cumulative Number Vaccinated"),
)

# %% Plot the antibody protection curve
pro_plot = pl.DataFrame({"ab": range(0, round(AB_SPIKE[1]))}).with_columns(
    risk=calculate_risk(
        pl.col("ab"), AB_RISK_SLOPE, AB_RISK_MIDPOINT, AB_RISK_MIN, AB_RISK_MAX
    )
)
alt.Chart(pro_plot).mark_line().encode(
    x=alt.X("ab:Q", title="Antibody Titer"),
    y=alt.Y("risk:Q", title="Risk", scale=alt.Scale(domain=[0, 1])),
)

# %% Collect a cross sectional serosurvey
xsec_samples = pl.DataFrame(
    {
        "id": np.random.choice(POP_SIZE, XSEC_SIZE, replace=False),
        "day": np.random.choice(
            np.arange(XSEC_WINDOW[0], XSEC_WINDOW[1]), XSEC_SIZE
        ),
    }
)

xsec_data = (
    all_data.join(xsec_samples, ["id", "day"], "semi")
    .select(["id", "ab"])
    .rename({"id": "pop_id", "ab": "titer"})
    .with_columns(pl.col("pop_id").cast(pl.UInt32), day=pl.lit("50-56"))
)

xsec_real = (
    all_data.filter((pl.col("day") >= 50) & (pl.col("day") <= 56))
    .select(["id", "ab", "day"])
    .rename({"id": "pop_id", "ab": "titer"})
    .with_columns(
        pl.col("pop_id").cast(pl.UInt32), pl.col("day").cast(pl.String)
    )
)

# %% Plot serosurvey titer distribution vs. true values
xsec_data_dens = sps.TiterSamples(
    xsec_data,
    pl.DataFrame({"titer": [0.0, 1000.0]}).with_row_index("pop_id"),
).to_density(values="titer", groups="day")

xsec_real_dens = sps.TiterSamples(
    xsec_real,
    pl.DataFrame({"titer": [0.0, 1000.0]}).with_row_index("pop_id"),
).to_density(values="titer", groups="day")

alt.data_transformers.disable_max_rows()
output = alt.Chart(xsec_real_dens).mark_line(
    opacity=0.1, color="green"
).encode(
    x=alt.X("titer:Q", title="Titer"),
    y=alt.Y("density:Q", title="Density"),
    detail="day",
) + alt.Chart(xsec_data_dens).mark_line(opacity=1.0, color="black").encode(
    x=alt.X("titer:Q"),
    y=alt.Y("density:Q"),
)
output.display()

# %% Conduct a TND serosurvey
tnd_inf_samples = all_data.filter(pl.col("inf_new")).sample(
    fraction=TND_INF_PRB
)
tnd_non_samples = all_data.filter(~pl.col("inf_status")).sample(
    fraction=TND_NON_PRB
)
tnd_data = pl.concat([tnd_inf_samples, tnd_non_samples])


# %% Build a numpyro model of protection
def dslogit_model(
    titer,
    infected,
    slope_shape=2.0,
    slope_rate=50,
    midpoint_shape=500.0,
    midpoint_rate=1.0,
    min_risk_shape1=1.0,
    min_risk_shape2=10.0,
    max_risk_shape1=10.0,
    max_risk_shape2=1.0,
):
    slope = numpyro.sample("slope", dist.Gamma(slope_shape, slope_rate))
    midpoint = numpyro.sample(
        "midpoint", dist.Gamma(midpoint_shape, midpoint_rate)
    )
    min_risk = numpyro.sample(
        "min_risk", dist.Beta(min_risk_shape1, min_risk_shape2)
    )
    max_risk = numpyro.sample(
        "max_risk", dist.Beta(max_risk_shape1, max_risk_shape2)
    )
    mu = min_risk + (max_risk - min_risk) / (
        1 + jnp.exp(slope * (titer - midpoint))
    )
    numpyro.sample("obs", dist.Binomial(probs=mu), obs=infected)
