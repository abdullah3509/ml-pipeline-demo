import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target], axis=1)
    return df

def preprocess_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=42)
