# %% Import modules
import numpy as np
import polars as pl

# %% Define parameters
POP_SIZE = 1000
NUM_DAYS = 100
RISK_SLOPE = 2.0
RISK_MIDPOINT = 120.0
RISK_MIN = 0.1
RISK_MAX = 0.9
AB_DECAY = 0.99
AB_SPIKE = 1000.0
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
                "titer": [0] * POP_SIZE,
            }
        )
    else:
        new_daily_data = daily_data[i - 1].with_columns(
            titer=pl.col("titer") * AB_DECAY
        )

    new_daily_data = (
        new_daily_data.with_columns(
            risk=calculate_risk(
                pl.col("titer"), RISK_SLOPE, RISK_MIDPOINT, RISK_MIN, RISK_MAX
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
            pl.when(
                (pl.col("inf_draw") < pl.col("risk")) & ~pl.col("inf_status")
            )
            .then(True)
            .otherwise(False)
            .alias("inf_new")
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
        .with_columns(
            titer=pl.when(pl.col("inf_new") | pl.col("vax_new"))
            .then(AB_SPIKE)
            .otherwise(pl.col("titer")),
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
        .select("id", "day", "inf_status", "vax_status", "titer")
    )

    daily_data.append(new_daily_data)

all_data = pl.concat(daily_data)
