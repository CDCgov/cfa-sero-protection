import polars as pl


def validate_schema(target_schema: dict, actual_schema: dict):
    """
    Check that columns of the correct names and types exist.
    """
    for name, type in target_schema.items():
        assert name in actual_schema, f"Column {name} is missing"
        assert type == actual_schema[name], (
            f"Column {name} is {actual_schema[name]} not {type}"
        )


def calculate_risk_dslogit(
    mid: pl.Expr,
    slope: pl.Expr,
    min: pl.Expr,
    max: pl.Expr,
    titer: pl.Expr,
):
    """
    Calculate risk from the 4 parameters of the double-scaled logit
    and an antibody titer.
    """
    return min + (max - min) / (1 + (slope * (titer - mid)).exp())


def calculate_protection_oddsratio(risk: pl.Expr):
    """
    Calculate protection using the odds ratio definition,
    using the greatest risk as the baseline for comparison.
    """
    return 1 - (risk / (1 - risk) / (risk.max() / (1 - risk.max())))
