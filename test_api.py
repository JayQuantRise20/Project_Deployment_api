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



def test_inference_status():
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

    # request the result from main 
    r = client.post("/predict",json.dumps(data))

    assert r.status_code==200


def test_inference_lowincome():
    data = {
        'age': 39,
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
    
    # request the result from main 
    r = client.post("/predict",json.dumps(data))

    assert  r.json()['inference_result']==' <=50K'


def test_inference_highincome():
    data = {
        'age': 31,
        'workclass':'Private',
        'fnlgt':45781,
        'education':'Masters',
        'education-num':14,
        'marital-status':'Never-married',
        'occupation':'Prof-specialty',
        'relationship':'Not-in-family',
        'race':'white',
        'sex':'Female',
        'capital-gain':14084,
        'capital-loss':0,
        'hours-per-week':50,
        'native-country':'United-States',
    }

    # request the result from main 
    r = client.post("/predict",json.dumps(data))

    assert  r.json()['inference_result']==' >50K'


