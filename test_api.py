import json 
from fastapi.testclient import TestClient
import pandas as pd
from joblib import load

from main import app
from starter.ml.data import process_data
from starter.ml.model import inference

client = TestClient(app)


def test_get_greeting_success():
    '''
    test for the greeting status code
    '''
    r = client.get('/')

    assert r.status_code==200

def test_get_greeting_result():
    '''
    test for the greeting message
    
    '''
    r = client.get('/')

    greeting = {'Greeting':'Welcome to the model deploymnet using Fast API'}

    assert r.json()==greeting


def test_inference_data():
    data = {
        'age':  39,
        'workclass':'State-gov',
        'fnlgt':77516,
        'education':'Bachelors',
        'education-num':13,
        'marital-status':'Never-married',
        'occupation':'Adm-clerical',
        'relationship':'Not-in-family',
        'race':'white',
        'sex':'Male',
        'capital-gain':2174,
        'capital-loss':0,
        'hours-per-week':40,
        'native-country':'United-States',
    }
    df = pd.DataFrame(data,index = [0])

    # import the model,lb and encoder 
    model = load('model/random_forest.joblib')
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')

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

    X,_,_,_ = process_data(df,categorical_features=cat_features,training=False,encoder=encoder, lb=lb)


    predected_salary = lb.inverse_transform(inference(model,X))
    result = {'inference_result':predected_salary}

    r = client.post("/data",json.dumps(data))

    print('r ',r.json())

    assert r.status_code==200







