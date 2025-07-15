# %% Import modules
import altair as alt
import numpy as np
import polars as pl

# %% Define parameters
POP_SIZE = 1000
NUM_DAYS = 100
AB_RISK_SLOPE = 0.02
AB_RISK_MIDPOINT = 500.0
AB_RISK_MIN = 0.1
AB_RISK_MAX = 0.9
AB_RISK_MIDPOINT_DROP = 2.0
AB_DECAY = [0.9, 1.0]
AB_SPIKE = [500.0, 1000.0]
TC_RISK_SLOPE = 0.02
TC_RISK_MIDPOINT = 500.0
TC_RISK_MIN = 0.1
TC_RISK_MAX = 0.9
TC_RISK_MIDPOINT_DROP = 2.0
TC_DECAY = [0.9, 1.0]
TC_SPIKE = [500.0, 1000.0]
FORCE_EXP = 0.1
FORCE_VAX = 0.02
RECOVERY = 0.25


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
                "ab": [0] * POP_SIZE,
                "tc": [0] * POP_SIZE,
            }
        )
    else:
        new_daily_data = daily_data[i - 1].with_columns(
            ab=pl.col("ab")
            * pl.Series(
                "ab",
                (np.random.rand(POP_SIZE) * (AB_DECAY[1] - AB_DECAY[0]))
                + AB_DECAY[0],
            ),
            tc=pl.col("tc")
            * pl.Series(
                "tc",
                (np.random.rand(POP_SIZE) * (TC_DECAY[1] - TC_DECAY[0]))
                + TC_DECAY[0],
            ),
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
            ab=pl.when(pl.col("inf_new") | pl.col("vax_new"))
            .then(
                pl.Series(
                    "ab",
                    (np.random.rand(POP_SIZE) * (AB_SPIKE[1] - AB_SPIKE[0]))
                    + AB_SPIKE[0],
                )
            )
            .otherwise(pl.col("ab")),
            tc=pl.when(pl.col("inf_new") | pl.col("vax_new"))
            .then(
                pl.Series(
                    "tc",
                    (np.random.rand(POP_SIZE) * (TC_SPIKE[1] - TC_SPIKE[0]))
                    + TC_SPIKE[0],
                )
            )
            .otherwise(pl.col("tc")),
        )
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
        .select("id", "day", "inf_status", "vax_status", "ab", "tc")
    )

    daily_data.append(new_daily_data)

all_data = pl.concat(daily_data)

# %% Plot number of people currently infected through time
inf_plot = all_data.group_by("day").agg(pl.col("inf_status").sum())
alt.Chart(inf_plot).mark_line().encode(
    x=alt.X("day:Q", title="Day"),
    y=alt.Y("inf_status:Q", title="Number Infected"),
)

# %% Plot number of people cumulatively vaccinated through time
vax_plot = all_data.group_by("day").agg(pl.col("vax_status").sum())
alt.Chart(vax_plot).mark_line().encode(
    x=alt.X("day:Q", title="Day"),
    y=alt.Y("vax_status:Q", title="Number Infected"),
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

# %%
