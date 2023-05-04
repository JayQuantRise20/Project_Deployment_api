# Put the code for your API here.
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel,Field
import json
import numpy as np

from starter.ml.data import process_data




app = FastAPI()

class IneferenceInput(BaseModel):
    age: int = Field(example=39)
    workclass:str=Field(example='State-gov')
    fnlgt:int = Field(example=77516)
    education:str = Field(example='Bachelors')
    education_num:int = Field(example=13)
    marital_status:str = Field(example='Never-married')
    occupation:str = Field(example='Adm-clerical')
    relationship:str = Field(example='Not-in-family')
    race:str = Field(example='white')
    sex:str = Field(example='Male')
    capital_gain:int = Field(example=2174)
    capital_loss:int = Field(example=0)
    hours_per_week:int = Field(example=40)
    native_country:str = Field(example='United-States')
    
    
@app.get('/')
def greeting():

    return {'Greeting':'Welcome to the model deploymnet using Fast API'}


@app.post('/data')
async def data_ingest(data:IneferenceInput):
      results = { "data": data}
      # convert data into numpy array as input for prediction
      #data_array = np.array(data.dict().values())
      # import the model and encoders
      
    return results





