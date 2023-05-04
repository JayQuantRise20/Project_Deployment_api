# Put the code for your API here.
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel,Field
import json
import numpy as np
import pandas as pd
from joblib import load

from starter.ml.data import process_data

from starter.ml.data import process_data
from starter.ml.model import inference
import requests
import os


app = FastAPI()

class IneferenceInput(BaseModel):
    age: int = Field(example=39)
    workclass:str=Field(example='State-gov')
    fnlgt:int = Field(example=77516)
    education:str = Field(example='Bachelors')
    education_num:int = Field(example=13,alias='education-num')
    marital_status:str = Field(example='Never-married',alias='marital-status')
    occupation:str = Field(example='Adm-clerical')
    relationship:str = Field(example='Not-in-family')
    race:str = Field(example='white')
    sex:str = Field(example='Male')
    capital_gain:int = Field(example=2174,alias='capital-gain')
    capital_loss:int = Field(example=0,alias='capital-loss')
    hours_per_week:int = Field(example=40,alias='hours-per-week')
    native_country:str = Field(example='United-States',alias='native-country')

class Result(BaseModel):
     inference_result:str
     
    

@app.get('/')
def greeting():
    
    return {'Greeting':'Welcome to the model deploymnet using Fast API'}

@app.post('/predict')
async def data_ingest(data:IneferenceInput):
      
      # import the model and encoders
      data_dict = {w.replace('_','-'):i for w,i in data.dict().items()}
      
      # convert the data to df to use ias input for processing 
      df = pd.DataFrame(data_dict,index = [0])

      # import the model,lb and encoder 
      model = load('model/random_forest.joblib')
      encoder = load('model/encoder.joblib')
      lb = load('model/lb.joblib')
      
      # categorical features 
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
      # process the data
      X,_,_,_ = process_data(df,categorical_features=cat_features,training=False,encoder=encoder, lb=lb)
      
      # predict the salary 
      predected_salary = lb.inverse_transform(inference(model,X))
      # save the results in dictionary 
      result = {'inference_result':predected_salary[0]}

      return result



     
     






