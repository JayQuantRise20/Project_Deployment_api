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
    
    
@app.get('/')
def greeting():
    
    return {'Greeting':'Welcome to the model deploymnet using Fast API'}


@app.post('/data')
async def data_ingest(data:IneferenceInput):
      # convert data into numpy array as input for prediction
      #data_array = np.array(data.dict().values())
      # import the model and encoders
      data_dict = data.dict()
      print('data dict' , data_dict['marital-status'])
      df = pd.DataFrame(data_dict,index = [0])
      print('df',df)

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
      print(resut)

      return result





