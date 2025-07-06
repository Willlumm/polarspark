from collections.abc import Sequence
from typing import Any, Self

import polars as pl

from polarspark.sql.column import Column
from polarspark.sql.dataframe import DataFrame


class GroupedData:
    def __init__(
        self,
        df: pl.DataFrame,
        group_by: Sequence[pl.Expr | str],
        pivot_on: str | None = None,
        pivot_values: list[Any] | None = None,
    ) -> None:
        self.df = df
        self.group_by = group_by
        self.pivot_on = pivot_on
        self.pivot_values = pivot_values

    def agg(self, *exprs: Column) -> DataFrame:
        if self.pivot_on is None:
            df = self.df.group_by(self.group_by).agg(*[column.expr for column in exprs])
        else:
            if not self.pivot_values:
                self.pivot_values = (
                    self.df.select(self.pivot_on).unique().to_series().to_list()
                )
            agg_names = [e.expr.meta.output_name() for e in exprs]
            rename_mapping = {
                f"{agg_name}_{pivot_value}": f"{pivot_value}_{agg_name}"
                for pivot_value in self.pivot_values
                for agg_name in agg_names
            }
            df = (
                self.df.filter(*[pl.col(self.pivot_on).is_in(self.pivot_values)])
                .group_by(*self.group_by, self.pivot_on)
                .agg(*[column.expr for column in exprs])
            )
            df = df.pivot(on=self.pivot_on, index=self.group_by).rename(rename_mapping)

        return DataFrame(df)

    def pivot(self, pivot_col: str, values: list[Any] | None = None) -> Self:
        cls = type(self)
        return cls(
            df=self.df, group_by=self.group_by, pivot_on=pivot_col, pivot_values=values
        )
