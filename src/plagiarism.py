import numpy as np
import pandas as pd
from build_index import Index
import pickle


class AntiPlag:
    def __init__(self):
        self._index = Index()

    def train_df(self, df: pd.DataFrame) -> None:
        pass

    def get_res(self, data: str, label: str) -> float:
        pass

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    ap = AntiPlag()
    ap.train_df(df)