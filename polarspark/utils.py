import polars as pl

from polarspark.sql import Column


def get_expr(col: Column | str) -> pl.Expr:
    if isinstance(col, Column):
        return col.expr
    if isinstance(col, str):
        return pl.col(col)
    raise TypeError(f"{type(col)}")
