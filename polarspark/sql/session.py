from collections.abc import Iterable
from typing import Any

import polars as pl
from pyspark.sql import Row
from pyspark.sql import types as T

from polarspark.converters import (
    convert_pyspark_schema_to_polars,
    convert_sql_schema_to_polars,
)
from polarspark.sql import DataFrame


class SparkSession:
    class Builder:
        def getOrCreate(self) -> "SparkSession":
            return SparkSession()

    def createDataFrame(
        self, data: Iterable[Any], schema: T.StructType | str | None = None
    ) -> DataFrame:
        if schema is None:
            pl_schema = None
        elif isinstance(schema, str):
            pl_schema = convert_sql_schema_to_polars(schema=schema)
        elif isinstance(schema, T.StructType):
            pl_schema = convert_pyspark_schema_to_polars(schema=schema)
        else:
            raise TypeError(f"{type(schema)}")

        if data and isinstance(data[0], Row):
            data = [row.asDict() for row in data]
        df = pl.DataFrame(data=data, schema=pl_schema, orient="row")
        return DataFrame(df=df)

    def stop(self) -> None:
        pass

    def table(self, tableName: str) -> DataFrame:
        pass
