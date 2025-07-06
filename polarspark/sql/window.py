from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from polarspark.sql import Column
from polarspark.sql import functions as F


def get_column(col: Column | str) -> Column:
    if isinstance(col, str):
        return F.col(col)
    return col


def get_columns(*cols: Column | str | Sequence[Column | str]) -> list[Column]:
    if all(isinstance(col, str) or isinstance(col, Column) for col in cols):
        return [get_column(col) for col in cols]
    if len(cols) == 1 and isinstance(cols[0], Sequence):
        return [get_column(col) for col in cols[0]]
    raise TypeError(f"{type(cols)}")


class WindowSpec:
    def __init__(self) -> None:
        self.partition_columns: list[Column] = []
        self.order_columns: list[Column] = []

    @property
    def partition_exprs(self) -> list[pl.Expr]:
        return [column.expr for column in self.partition_columns]

    @property
    def order_exprs(self) -> list[pl.Expr]:
        return [column.expr for column in self.order_columns]

    def partitionBy(self, *cols: Column | str | Sequence[Column | str]) -> WindowSpec:
        self.partition_columns = get_columns(*cols)
        return self

    def orderBy(self, *cols: Column | str | Sequence[Column | str]) -> WindowSpec:
        self.order_columns = get_columns(*cols)
        return self


class Window:
    @staticmethod
    def partitionBy(*cols: Column | str | Sequence[Column | str]) -> WindowSpec:
        return WindowSpec().partitionBy(*cols)
