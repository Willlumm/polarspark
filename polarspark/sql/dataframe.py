from collections.abc import Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, Self

import polars as pl
from pyspark.sql import Row
from pyspark.sql import types as T

from polarspark.converters import convert_polars_schema_to_pyspark
from polarspark.sql import Column

if TYPE_CHECKING:
    from polarspark.sql.group import GroupedData


class DataFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    @property
    def columns(self) -> list[str]:
        return self.df.columns

    @property
    def schema(self) -> T.StructType:
        return convert_polars_schema_to_pyspark(schema=self.df.schema)

    def agg(self, *exprs: Column) -> Self:
        return self.select(*exprs)

    def collect(self) -> list[Row]:
        return [Row(**row) for row in self.df.rows(named=True)]

    def distinct(self) -> Self:
        cls = type(self)
        df = self.df.unique()
        return cls(df)

    def dropDuplicates(self, subset: list[str]) -> Self:
        cls = type(self)
        df = self.df.unique(subset=subset)
        return cls(df)

    def dropna(self, subset: str | Sequence[str]) -> Self:
        cls = type(self)
        df = self.df.drop_nulls(subset=subset)
        return cls(df)

    def fillna(self, value: Any, subset: str | Sequence[str] | None = None) -> Self:
        if subset is None:
            subset = self.columns
        elif isinstance(subset, str):
            subset = [subset]
        cls = type(self)
        df = self.df.with_columns(
            *[pl.col(column).fill_null(value) for column in subset]
        )
        return cls(df)

    def filter(self, condition: Column | str) -> Self:
        cls = type(self)
        if isinstance(condition, str):
            df = self.df.sql(f"SELECT * FROM self WHERE {condition}")
        elif isinstance(condition, Column):
            df = self.df.filter(condition.expr)
        return cls(df)

    def groupBy(self, *cols: Column | str) -> "GroupedData":
        exprs = [col.expr if isinstance(col, Column) else col for col in cols]
        from polarspark.sql.group import GroupedData

        return GroupedData(self.df, exprs)

    def join(self, other: Self, on: str, how: str):
        cls = type(self)
        df = self.df.join(other=other.df, on=on, how=how)
        return cls(df)

    def select(self, *cols: Column | str) -> Self:
        cls = type(self)
        df = self.df.select(
            *[col.expr if isinstance(col, Column) else col for col in cols]
        )
        return cls(df)

    def where(self, condition: Column) -> Self:
        return self.filter(condition)

    def withColumn(self, colName: str, col: Column) -> Self:
        cls = type(self)
        df = self.df.with_columns(col.expr.alias(colName))
        return cls(df)

    def withColumns(self, colsMap: dict[str, Column]) -> Self:
        cls = type(self)
        output_names = set(
            chain.from_iterable(
                column.expr.meta.root_names() for column in colsMap.values()
            )
        )
        if colsMap.keys() & output_names:
            df = self.df
            for name, column in colsMap.items():
                df = df.with_columns(**{name: column.expr})
        else:
            df = self.df.with_columns(
                **{name: column.expr for name, column in colsMap.items()}
            )
        return cls(df)

    def withColumnRenamed(self, existing: str, new: str) -> Self:
        cls = type(self)
        df = self.df.rename({existing: new})
        return cls(df)
