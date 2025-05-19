import polars as pl


def validate_schema(target: dict, actual: dict):
    """
    Check that columns of the correct names and types exist.
    """
    for name, type in target.items():
        assert name in actual, f"Column {name} is missing"
        assert type == actual[name], f"{name} is {actual[name]} not {type}"


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


def calculate_protection_oddsratio(risk: pl.Expr, max_risk: pl.Expr):
    """
    Calculate protection using the odds ratio definition,
    using the greatest risk as the baseline for comparison.
    """
    return 1 - (risk / (1 - risk) / (max_risk / (1 - max_risk)))
