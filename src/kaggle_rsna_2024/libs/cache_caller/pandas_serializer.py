import pandas as pd

class PandasSerializer:
    def serialize(self, df, filename):
        df.to_csv(filename)

    def deserialize(self, filename):
        return pd.read_csv(filename)