from __future__ import annotations
from typing import overload

import polars as pl
import polars.testing
import pyspark.sql
import pyspark.testing

import polarspark.sql


@overload
def assertDataFrameEqual(
    actual: polarspark.sql.DataFrame,
    expected: polarspark.sql.DataFrame,
    checkRowOrder: bool = False,
) -> None: ...


@overload
def assertDataFrameEqual(
    actual: pl.DataFrame, expected: pl.DataFrame, checkRowOrder: bool = False
) -> None: ...


@overload
def assertDataFrameEqual(
    actual: pyspark.sql.DataFrame,
    expected: pyspark.sql.DataFrame,
    checkRowOrder: bool = False,
) -> None: ...


def assertDataFrameEqual(
    actual: polarspark.sql.DataFrame | pl.DataFrame | pyspark.sql.DataFrame,
    expected: polarspark.sql.DataFrame | pl.DataFrame | pyspark.sql.DataFrame,
    checkRowOrder: bool = False,
) -> None:
    if isinstance(actual, polarspark.sql.DataFrame) and isinstance(
        expected, polarspark.sql.DataFrame
    ):
        polars.testing.assert_frame_equal(
            left=actual.df, right=expected.df, check_row_order=checkRowOrder
        )
        return

    if isinstance(actual, pl.DataFrame) and isinstance(expected, pl.DataFrame):
        polars.testing.assert_frame_equal(
            left=actual, right=expected, check_row_order=checkRowOrder
        )
        return

    if isinstance(actual, pyspark.sql.DataFrame) and isinstance(
        expected, pyspark.sql.DataFrame
    ):
        pyspark.testing.assertDataFrameEqual(
            actual=actual, expected=expected, checkRowOrder=checkRowOrder
        )
        return

    message = (
        "Dataframes must be the same type."
        f" Got actual: {type(actual)}, expected: {type(expected)}."
    )
    raise TypeError(message)
