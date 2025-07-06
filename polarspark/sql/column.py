from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import polars as pl
from polars.expr.whenthen import ChainedThen, Then
from pyspark.sql import types as T

from polarspark.converters import convert_pyspark_type_to_polars

if TYPE_CHECKING:
    from polarspark.sql.window import WindowSpec


class Column:
    def __init__(self, expr: pl.Expr, order_descending: bool = False) -> None:
        self.expr = expr
        self._order_descending = order_descending

    def __and__(self, value: object) -> "Column":
        cls = type(self)
        if isinstance(value, cls):
            return cls(self.expr & value.expr)
        return cls(self.expr & pl.lit(value))

    def __eq__(self, value: object) -> "Column":  # type: ignore[override]
        cls = type(self)
        if isinstance(value, cls):
            return cls(self.expr == value.expr)
        return cls(self.expr == pl.lit(value))

    def __ge__(self, value: object) -> "Column":
        cls = type(self)
        if isinstance(value, cls):
            return cls(self.expr >= value.expr)
        return cls(self.expr >= pl.lit(value))

    def __invert__(self) -> "Column":
        cls = type(self)
        return cls(~self.expr)

    def __ne__(self, value: object) -> "Column":  # type: ignore[override]
        cls = type(self)
        if isinstance(value, cls):
            return cls(self.expr != value.expr)
        return cls(self.expr != pl.lit(value))

    def __or__(self, value: object) -> "Column":
        cls = type(self)
        if isinstance(value, cls):
            return cls(self.expr | value.expr)
        return cls(self.expr | pl.lit(value))

    def alias(self, alias: str) -> "Column":
        cls = type(self)
        expr = self.expr.alias(name=alias)
        return cls(expr)

    def asc(self) -> "Column":
        cls = type(self)
        return cls(self.expr, order_descending=False)

    def cast(self, dataType: T.DataType) -> "Column":
        cls = type(self)
        expr = self.expr.cast(convert_pyspark_type_to_polars(data_type=dataType))
        return cls(expr)

    def desc(self) -> "Column":
        cls = type(self)
        return cls(self.expr, order_descending=True)

    def isin(self, *cols: Any) -> "Column":
        cls = type(self)
        if (
            len(cols) == 1
            and isinstance(cols[0], Sequence)
            and not isinstance(cols[0], str)
        ):
            expr = self.expr.is_in(cols[0])
        else:
            expr = self.expr.is_in(cols)
        return cls(expr)

    def isNotNull(self) -> "Column":
        cls = type(self)
        expr = self.expr.is_not_null()
        return cls(expr)

    def isNull(self) -> "Column":
        cls = type(self)
        expr = self.expr.is_null()
        return cls(expr)

    def otherwise(self, value: Any) -> "Column":
        if not (isinstance(self.expr, Then) or isinstance(self.expr, ChainedThen)):
            raise AttributeError("`otherwise()` can only be used after `when()`")
        if isinstance(value, Column):
            value = value.expr
        cls = type(self)
        expr = self.expr.otherwise(value)
        return cls(expr)

    def over(self, window: "WindowSpec") -> "Column":
        cls = type(self)
        expr = self.expr.over(
            partition_by=window.partition_exprs,
            order_by=window.order_exprs,
            descending=window.order_columns[0]._order_descending,
        )
        return cls(expr)

    def when(self, condition: "Column", value: Any) -> "Column":
        if not (isinstance(self.expr, Then) or isinstance(self.expr, ChainedThen)):
            raise AttributeError("`when()` can only be used after `then()`")
        if isinstance(value, Column):
            value = value.expr
        cls = type(self)
        expr = self.expr.when(condition.expr).then(value)
        return cls(expr)
