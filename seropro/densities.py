import altair as alt
import polars as pl

import seropro.utils


class Density(pl.DataFrame):
    """
    Estimated density of any distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def plot(self):
        id_col = [col for col in self.columns if "_id" in col]
        if id_col == []:
            assert self.shape[1] == 2, (
                "The Density object has >2 columns but no id column."
            )
            x_col = (set(self.columns) - {"density"}).pop()
            output = (
                alt.Chart(self)
                .mark_line(opacity=1.0, color="black")
                .encode(
                    x=x_col + ":Q",
                    y="density:Q",
                )
            )
        else:
            assert self.shape[1] == 3, (
                "The Density object has >2 columns beyond the id column."
            )
            alt.data_transformers.disable_max_rows()
            id_col = id_col[0]
            x_col = (set(self.columns) - {"density", id_col}).pop()
            mean_density = (
                self.drop(id_col)
                .group_by(x_col)
                .agg(density=pl.col("density").mean())
            )
            output = alt.Chart(self).mark_line(
                opacity=0.1, color="green"
            ).encode(
                x=alt.X(x_col + ":Q", title=x_col),
                y=alt.Y("density:Q", title="Density"),
                detail=id_col,
            ) + alt.Chart(mean_density).mark_line(
                opacity=1.0, color="black"
            ).encode(
                x=x_col + ":Q",
                y="density:Q",
            )
        output.display()


class PopDensity(Density):
    """
    Estimated density of a distribution of individuals in a population.
    """

    def validate(self):
        raise NotImplementedError("Subclasses must implement this method.")


class ParDensity(Density):
    """
    Estimated density of a distribution of posterior draws.
    """

    def validate(self):
        raise NotImplementedError("Subclasses must implement this method.")


class TiterPopDensity(PopDensity):
    """
    Estimated density of a distribution of antibody titers of individuals in a population.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        seropro.utils.validate_schema(
            {"titer": pl.Float64, "density": pl.Float64}, self.schema
        )


class RiskPopDensity(PopDensity):
    """
    Estimated density of a distribution of risks of individuals in a population,
    repeated for many draws from a risk curve's posterior distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        seropro.utils.validate_schema(
            {"risk": pl.Float64, "density": pl.Float64, "par_id": pl.Int64},
            self.schema,
        )


class ProtectionPopDensity(PopDensity):
    """
    Estimated density of a distribution of protections of individuals in a population,
    repeated for many draws from a risk curve's posterior distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        seropro.utils.validate_schema(
            {
                "protection": pl.Float64,
                "density": pl.Float64,
                "par_id": pl.Int64,
            },
            self.schema,
        )
