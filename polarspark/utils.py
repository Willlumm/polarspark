from __future__ import annotations

import polars as pl

from polarspark.sql import Column


def get_expr(col: Column | str) -> pl.Expr:
    """Convert a Column or column name to a Polars expression."""
    if isinstance(col, Column):
        return col.expr
    if isinstance(col, str):
        return pl.col(col)
    message = f"{type(col)}"
    raise TypeError(message)
