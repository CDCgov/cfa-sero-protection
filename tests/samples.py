import polars as pl
import polars.testing as plt
import pytest

import seropro.samples as sps
import seropro.utils as spu


@pytest.fixture
def titers():
    """
    Mock titer data set for testing.
    """
    titers = sps.TiterSamples(
        pl.DataFrame({"titer": [0.0, 10.0, 20.0]}).with_row_index("pop_id"),
        pl.DataFrame({"titer": [0.0, 30.0]}).with_row_index("pop_id"),
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
                "slope": [1.0, 10.0, 20.0],
                "min": [0.0, 0.1, 0.2],
                "max": [1.0, 0.9, 0.8],
            }
        )
        .with_row_index("par_id")
        .unpivot(index="par_id", variable_name="par", value_name="val"),
        None,
    )

    return curves


def test_to_risk_right_func(titers, curves):
    """
    When given sensible inputs, to_risk should compute correctly.
    """
    output = titers.to_risk(curves, spu.calculate_risk_dslogit)

    output.draws = output.draws.with_columns(risk=pl.col("risk").round(2))
    output.bounds = output.bounds.with_columns(risk=pl.col("risk").round(2))

    expected_draws = pl.DataFrame(
        {
            "pop_id": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "par_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "risk": [1.0, 0.5, 0.0, 0.9, 0.9, 0.5, 0.5, 0.2, 0.2],
        }
    ).with_columns(
        pop_id=pl.col("pop_id").cast(pl.UInt32),
        par_id=pl.col("par_id").cast(pl.UInt32),
    )

    plt.assert_frame_equal(
        output.draws,
        expected_draws,
        check_row_order=False,
        check_column_order=False,
    )

    expected_bounds = pl.DataFrame(
        {
            "pop_id": [0, 1, 0, 1, 0, 1],
            "par_id": [0, 0, 1, 1, 2, 2],
            "risk": [1.0, 0.0, 0.9, 0.1, 0.5, 0.2],
        }
    ).with_columns(
        pop_id=pl.col("pop_id").cast(pl.UInt32),
        par_id=pl.col("par_id").cast(pl.UInt32),
    )

    plt.assert_frame_equal(
        output.bounds,
        expected_bounds,
        check_row_order=False,
        check_column_order=False,
    )


def test_to_risk_wrong_func(titers, curves):
    """
    When given a risk functions whose arguments do not match the parameters
    available in the CurveSamples, to_risk raises the right error.
    """
    with pytest.raises(AssertionError):
        titers.to_risk(curves, spu.calculate_protection_oddsratio)


def test_to_density_no_groups(titers):
    """
    When there is only one set of sample ids, to_density calculates a density
    over all samples.
    """
    output = titers.to_density("titer", None)
    assert output.shape == (1000, 2)
    spu.validate_schema(
        {"titer": pl.Float64, "density": pl.Float64}, output.schema
    )


def test_to_density_groups(titers, curves):
    """
    When there are two sets of sample ids, to_density calculates multiple densities:
    one over all pop_ids for each par_id.
    """
    output = titers.to_risk(curves, spu.calculate_risk_dslogit).to_density(
        "risk", "par_id"
    )
    assert output.shape == (3000, 3)
    spu.validate_schema(
        {"risk": pl.Float64, "density": pl.Float64, "par_id": pl.UInt32},
        output.schema,
    )


def test_to_protection_right_func(titers, curves):
    """
    When given a protection functions that only requires a risk argument,
    return protection as expected.
    """
    output = titers.to_risk(curves, spu.calculate_risk_dslogit).to_protection(
        spu.calculate_protection_oddsratio
    )

    output.draws = output.draws.with_columns(
        protection=pl.col("protection").round(2)
    )
    output.bounds = output.bounds.with_columns(
        protection=pl.col("protection").round(2)
    )

    expected_draws = pl.DataFrame(
        {
            "pop_id": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "par_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "protection": [0.0, 1.0, 1.0, 0.0, 0.0, 0.89, 0.0, 0.75, 0.75],
        }
    ).with_columns(
        pop_id=pl.col("pop_id").cast(pl.UInt32),
        par_id=pl.col("par_id").cast(pl.UInt32),
    )

    expected_bounds = pl.DataFrame(
        {
            "pop_id": [0, 1, 0, 1, 0, 1],
            "par_id": [0, 0, 1, 1, 2, 2],
            "protection": [0.0, 1.0, 0.0, 0.99, 0.0, 0.75],
        }
    ).with_columns(
        pop_id=pl.col("pop_id").cast(pl.UInt32),
        par_id=pl.col("par_id").cast(pl.UInt32),
    )

    plt.assert_frame_equal(
        output.draws,
        expected_draws,
        check_row_order=False,
        check_column_order=False,
    )

    plt.assert_frame_equal(
        output.bounds,
        expected_bounds,
        check_row_order=False,
        check_column_order=False,
    )


def test_to_protection_wrong_func(titers, curves):
    """
    When given a protection functions that requires more than one argument
    or an argument not called risk, raise the right error.
    """
    with pytest.raises(AssertionError):
        titers.to_risk(curves, spu.calculate_risk_dslogit).to_protection(
            spu.calculate_risk_dslogit
        )
