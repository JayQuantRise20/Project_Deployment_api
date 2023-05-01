import pandas as pd
import pytest
import numpy as np
from joblib import load

from starter.ml.data import process_data

@pytest.fixture
def data():
    """
    Function to read the data for testing on functions 
    """
    df = pd.read_csv("./data/census.csv")

    return df


def test_data_shape(data):
    '''
    test the shape of data after droping ht enull values to assure that 
    there is no null value in the data
    '''

    assert data.shape == data.dropna().shape


def test_categorical(data):
    '''
    test to evaluate number of cateorical values match with the encoded array shape
    '''
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    categorical_data = data[cat_features]

    num_of_unique_columns = []
    for cat in categorical_data.columns:
        num_of_unique_columns.append(data[cat].unique())

    # flatten the list num_of_unique_columns
    num_of_unique_columns = [item for sublist in num_of_unique_columns for item in sublist]

    # load the encoder 
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    # find the columns of in encoder
    encoded_data = encoder.transform(categorical_data.values)

    assert len(num_of_unique_columns) == encoded_data.shape[1]


def test_labels(data):
    '''
    test to evaluate number of  labels match with the encoded array 
    '''
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    labels = data['salary']
    unique_labels = labels.unique()

    # load the encoder 
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    # load the lb 
    # transform y values
    y = np.unique(lb.transform(labels.values).ravel())
    
    assert unique_labels.shape[0] == y.shape[0]



