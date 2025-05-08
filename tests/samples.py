import polars as pl
import polars.testing as plt
import pytest

import seropro.densities as sds
import seropro.samples as sps
import seropro.utils as spu


@pytest.fixture
def titers():
    """
    Mock titer data set for testing.
    """
    titers = sps.TiterSamples(
        pl.DataFrame({"titer": [0.0, 10.0, 20.0]}).with_row_index("pop_id")
    )

    return titers


@pytest.fixture
def curves():
    """
    Mock curve posterior set for testing.
    """
    curves = sps.CurveSamples(
        pl.DataFrame(
            {
                "mid": [10.0, 20.0, 0.0],
                "slope": [0.0, 10.0, 20.0],
                "min": [0.0, 0.1, 0.2],
                "max": [1.0, 0.9, 0.8],
            }
        )
        .with_row_index("par_id")
        .unpivot(index="par_id", variable_name="par", value_name="val")
    )

    return curves


def test_to_risk_right_func(titers, curves):
    """
    When given sensible inputs, to_risk should compute correctly.
    """
    output = titers.to_risk(curves, spu.calculate_risk_dslogit).with_columns(
        risk=pl.col("risk").round(2)
    )

    expected = pl.DataFrame(
        {
            "pop_id": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "par_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "risk": [0.5, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.2, 0.2],
        }
    ).with_columns(
        pop_id=pl.col("pop_id").cast(pl.UInt32),
        par_id=pl.col("par_id").cast(pl.UInt32),
    )

    plt.assert_frame_equal(
        output, expected, check_row_order=False, check_column_order=False
    )


def test_to_risk_wrong_func(titers, curves):
    """
    When given a risk functions whose arguments do not match the parameters
    available in the CurveSamples, to_risk raises the right error.
    """
