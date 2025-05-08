import polars as pl
import pytest

import seropro.samples as sps


@pytest.fixture
def titers():
    """
    Mock titer data set for testing.
    """
    titers = sps.TiterSamples(
        pl.DataFrame({"titer": [10, 11, 12, 20, 21, 22]}).with_row_index(
            "pop_id"
        )
    )

    return titers


@pytest.fixture
def curve():
    """
    Mock curve posterior set for testing.
    """
    curve = sps.CurveSamples(
        pl.DataFrame(
            {
                "midpoint": [14, 15, 16],
                "slope": [9, 10, 11],
                "min": [0.1, 0.15, 0.2],
                "max": [0.8, 0.85, 0.9],
            }
        )
        .with_row_index("par_id")
        .unpivot(variable_name="par", value_name="val")
    )

    return curve
