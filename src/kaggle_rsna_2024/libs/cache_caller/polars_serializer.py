import polars as pl

class PolarsSerializer:
    def serialize(self, df, filename):
        df.write_parquet(filename)

    def deserialize(self, filename):
        return pl.read_parquet(filename)