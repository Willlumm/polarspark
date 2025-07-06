from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import polars as pl
from pyspark.sql import types as T

TypeSource = Literal["sql", "polars", "pyspark"]
DataType = Union[pl.DataType, T.DataType, str]


@dataclass(frozen=True)
class DataTypeFamily:
    sql: str
    polars: pl.DataType
    pyspark: T.DataType


_DATA_TYPES = [
    DataTypeFamily("BOOLEAN", pl.Boolean(), T.BooleanType()),
    DataTypeFamily("TINYINT", pl.Int8(), T.ByteType()),
    DataTypeFamily("SMALLINT", pl.Int16(), T.ShortType()),
    DataTypeFamily("INT", pl.Int32(), T.IntegerType()),
    DataTypeFamily("BIGINT", pl.Int64(), T.LongType()),
    DataTypeFamily("DATE", pl.Date(), T.DateType()),
    DataTypeFamily("TIMESTAMP", pl.Datetime(), T.TimestampType()),
    DataTypeFamily("STRING", pl.String(), T.StringType()),
]


_POLARS_TO_PYSPARK = {data_type.polars: data_type.pyspark for data_type in _DATA_TYPES}
_PYSPARK_TO_POLARS = {data_type.pyspark: data_type.polars for data_type in _DATA_TYPES}
_SQL_TO_POLARS = {data_type.sql: data_type.polars for data_type in _DATA_TYPES}


def convert_polars_type_to_pyspark(data_type: pl.DataType) -> T.DataType:
    return _POLARS_TO_PYSPARK[data_type]


def convert_pyspark_type_to_polars(data_type: T.DataType) -> pl.DataType:
    return _PYSPARK_TO_POLARS[data_type]


def convert_sql_type_to_polars(data_type: str) -> pl.DataType:
    return _SQL_TO_POLARS[data_type.upper()]


def convert_polars_schema_to_pyspark(schema: pl.Schema) -> T.StructType:
    return T.StructType(
        [
            T.StructField(
                name=name, dataType=convert_polars_type_to_pyspark(data_type=data_type)
            )
            for name, data_type in schema.items()
        ]
    )


def convert_pyspark_schema_to_polars(schema: T.StructType) -> pl.Schema:
    return pl.Schema(
        {
            field.name: convert_pyspark_type_to_polars(data_type=field.dataType)
            for field in schema.fields
        }
    )


def convert_sql_schema_to_polars(schema: str) -> pl.Schema:
    fields = {}
    for field in schema.split(","):
        name, data_type = field.split()
        fields[name] = convert_sql_type_to_polars(data_type=data_type)
    return pl.Schema(fields)
