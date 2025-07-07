import pytest
from polarspark.sql import SparkSession


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["spark"] = SparkSession.Builder().getOrCreate()