from typing import Any

import polars as pl

from polarspark.sql import Column
from polarspark.utils import get_expr


def coalesce(*cols: Column | str) -> Column:
    expr = pl.coalesce(get_expr(col) for col in cols)
    return Column(expr)


def col(col: str) -> Column:
    expr = pl.col(name=col)
    return Column(expr)


def first(col: Column | str, ignorenulls: bool = False) -> Column:
    expr = get_expr(col)
    if ignorenulls:
        expr = expr.drop_nulls()
    expr = expr.first()
    return Column(expr)


def greatest(*cols: Column | str) -> Column:
    expr = pl.max_horizontal(get_expr(col) for col in cols)
    return Column(expr)


def lit(col: Any) -> Column:
    expr = pl.lit(value=col)
    return Column(expr)


def max(col: Column | str) -> Column:
    expr = get_expr(col).max()
    return Column(expr)


def min(col: Column | str) -> Column:
    expr = get_expr(col).min()
    return Column(expr)


def regexp_replace(string: Column | str, pattern: str, replacement: str) -> Column:
    expr = get_expr(string).str.replace_all(pattern=pattern, value=replacement)
    return Column(expr)


def row_number() -> Column:
    expr = pl.int_range(start=1, end=pl.len() + 1)
    return Column(expr)


def to_date(col: Column | str, format: str | None = None) -> Column:
    expr = get_expr(col).str.to_date(format=format, strict=False)
    return Column(expr)


def trim(col: Column | str, trim: str | None = None) -> Column:
    expr = get_expr(col).str.strip_chars(characters=trim)
    return Column(expr)


def when(condition: Column, value: Any) -> Column:
    expr = pl.when(condition.expr)
    if isinstance(value, Column):
        value = value.expr
    expr = expr.then(value)
    return Column(expr)
