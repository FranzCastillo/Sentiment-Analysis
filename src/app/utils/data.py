import pandas as pd
import joblib

class Data:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.model = joblib.load('data/model.pkl')

    def predict(self, X):
        return self.model.predict(X)
