import pytest
from src.utils import load_data, preprocess_data

def test_load_data_shape():
    df = load_data()
    assert df.shape[0] > 0  # Data is not empty

def test_preprocess_data():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

def test_model_accuracy():
    from src.ml_pipeline import model, X_test, y_test
    acc = model.score(X_test, y_test)
    assert acc > 0.8  # Model accuracy > 80%
