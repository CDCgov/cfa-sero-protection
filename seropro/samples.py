from typing import Callable, Dict, List

import altair as alt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

import seropro.utils
from seropro.densities import Density


class Samples(pl.DataFrame):
    """
    Samples drawn from any distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def to_density(
        self, values: str, groups: str | None, bins: int = 1000
    ) -> Density:
        x_vals = np.linspace(
            np.array(self[values].min()), np.array(self[values].max()), bins
        )
        if groups is None:
            kde = gaussian_kde(self[values].to_numpy())
            density = pl.DataFrame({values: x_vals, "density": kde(x_vals)})
        else:
            density = pl.DataFrame()
            for i in self[groups].unique():
                group = self.filter(pl.col(groups) == i)
                kde = gaussian_kde(group[values].to_numpy())
                group_density = pl.DataFrame(
                    {
                        values: x_vals,
                        "density": kde(x_vals),
                        groups: i,
                    }
                ).with_columns(pl.col(groups).cast(pl.Int64).alias(groups))
                density = density.vstack(group_density)

        return Density(density)


class TiterSamples(Samples):
    """
    Antibody titer samples drawn from a distribution of individuals in a population.
    """

    def validate(self):
        seropro.utils.validate_schema(
            {"titer": pl.Float64, "pop_id": pl.UInt32}, self.schema
        )
        assert (self["titer"] >= 0.0).all(), "Some titers <0."

    def to_risk(
        self, curves: "CurveSamples", risk_func: Callable
    ) -> "RiskSamples":
        par_samples = curves.pivot(on="par", values="val")
        col_names = risk_func.__code__.co_varnames[
            : risk_func.__code__.co_argcount
        ]
        risk_samples = self.join(par_samples, how="cross")
        cols = tuple(pl.col(name) for name in col_names)
        risk_samples = risk_samples.with_columns(risk=risk_func(*cols)).select(
            ["risk", "pop_id", "par_id"]
        )

        return RiskSamples(risk_samples)


class CurveSamples(Samples):
    """
    Samples drawn from a posterior distribution of parameter values of a risk curve.
    """

    def validate(self):
        seropro.utils.validate_schema(
            {"par": pl.String, "val": pl.Float64, "par_id": pl.UInt32},
            self.schema,
        )

    def plot(
        self,
        risk_func: Callable,
        titer_min: float = 0,
        titer_max: float = 1000,
        bins: int = 1000,
    ):
        titers = TiterSamples(
            pl.DataFrame(
                {
                    "titer": np.linspace(titer_min, titer_max, bins),
                }
            ).with_row_index("pop_id")
        )
        risks = titers.to_risk(self, risk_func).join(titers, on="pop_id")
        alt.data_transformers.disable_max_rows()
        mean_risk = risks.group_by("titer").agg(risk=pl.col("risk").mean())
        output = alt.Chart(risks).mark_line(opacity=0.1).encode(
            x=alt.X("titer:Q", title="Titer"),
            y=alt.Y("risk:Q", title="Risk"),
            detail="par_id",
        ) + alt.Chart(mean_risk).mark_line(opacity=1.0, color="black").encode(
            x=alt.X("titer:Q"),
            y=alt.Y("risk:Q"),
        )
        output.display()


class RiskSamples(Samples):
    """
    Samples of risk drawn from a distribution of individuals in a population
    and from a posterior distribution of parameter values.
    """

    def validate(self):
        seropro.utils.validate_schema(
            {"risk": pl.Float64, "pop_id": pl.UInt32, "par_id": pl.UInt32},
            self.schema,
        )
        assert (self["risk"] >= 0.0).all(), "Some risks <0."
        assert (self["risk"] <= 1.0).all(), "Some risks >1."

    def to_protection(self, prot_func: Callable) -> "ProtectionSamples":
        prot_samples = self.with_columns(
            protection=prot_func(pl.col("risk"))
        ).drop("risk")

        return ProtectionSamples(prot_samples)


class ProtectionSamples(Samples):
    """
    Samples of protection drawn from a distribution of individuals in a population
    and from a posterior distribution of parameter values.
    """

    def validate(self):
        seropro.utils.validate_schema(
            {
                "protection": pl.Float64,
                "pop_id": pl.UInt32,
                "par_id": pl.UInt32,
            },
            self.schema,
        )
        assert (self["protection"] >= 0.0).all(), "Some protections <0."
        assert (self["protection"] <= 1.0).all(), "Some protections >1."


def simulate_titers(dists: List[tuple], seed: int = 0):
    """
    Simulate titers by sampling from an arbitrary mixture of distributions,
    specified in a list of tuples that contain the name of each distribution,
    the parameter values of that distribution, and the number of draws.

    This is TEMPORARY until genuine simulations or data are available.
    """
    rng = np.random.default_rng(seed)
    titers = []
    for dist in dists:
        assert type(dist[0]) is str, "A distribution is not named."
        assert hasattr(rng, dist[0]), "{dist[0]} is not a distribution."
        titers = titers + getattr(rng, dist[0])(*dist[1:]).tolist()
    titer_samples = pl.DataFrame({"titer": titers}).with_row_index("pop_id")
    return TiterSamples(titer_samples)


def simulate_curve(dists: Dict[str, tuple], seed: int = 0):
    """
    Simulate parameter draws for a risk curve, specified in a dictionary where keys
    are the name of the curve parameter, and values are lists of the name of the
    distribution to draw from, its parameter values, and the number of draws.
    Correlation structures aren't supported.

    The shape of the desired risk curve is also provided.

    This is TEMPORARY until genuine simulations or data are availalble.
    """
    rng = np.random.default_rng(seed)
    draws = pl.DataFrame()
    for name, dist in dists.items():
        assert type(dist[0]) is str, "A distribution is not named."
        assert hasattr(rng, dist[0]), "{dist} is not a distribution."
        draws = draws.hstack(
            pl.DataFrame({name: getattr(rng, dist[0])(*dist[1:]).tolist()})
        )
    draws = draws.with_row_index("par_id").unpivot(
        index="par_id", variable_name="par", value_name="val"
    )
    return CurveSamples(draws)
