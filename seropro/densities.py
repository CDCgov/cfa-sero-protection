import altair as alt
import polars as pl


class Density(pl.DataFrame):
    """
    Estimated density of any distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self):
        id_col = [col for col in self.columns if "_id" in col]
        if id_col == []:
            assert self.shape[1] == 2, ">2 columns but no id column."
            x_col = (set(self.columns) - {"density"}).pop()
            output = (
                alt.Chart(self)
                .mark_line(opacity=1.0, color="black")
                .encode(
                    x=alt.X(x_col + ":Q"),
                    y=alt.Y("density:Q"),
                )
            )
        else:
            assert self.shape[1] == 3, ">2 columns beyond the id column."
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
                x=alt.X(x_col + ":Q"),
                y=alt.Y("density:Q"),
            )
        output.display()
