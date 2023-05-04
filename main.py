# Put the code for your API here.
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel,Field
import json

from starter.ml.data import process_data




app = FastAPI()

class IneferenceInput(BaseModel):
    age: int = Field(example=39)
    workclass:str=Field(example='State-gov')
    fnlgt:int = Field(example=77516)
    education:str = Field(example='Bachelors')
    education-num:int = Field(example=13)
    marital-status:str = Field(example='Never-married')
    occupation:str = Field(example='Adm-clerical')
    relationship:str = Field(example='Not-in-family')
    race:str = Field(example='white')
    sex:str = Field(example='Male')
    capital-gain:int = Field(example=2174)
    capital-loss:int = Field(example=0)
    hours-per-week:int = Field(example=40)
    native-country:str = Field(example='United-States')
    
    
@app.get('/')
def greeting():

    return {'Greeting':'welcome to the model deploymnet using Fast API'}


@app.post('/predict/{item_id}')
async def predict(item_id:int, data: IneferenceInput):
      results = {"item_id": item_id, "data": data}
      # convert data into numpy array as input for prediction
      data_array = np.array(data.dict().values())
      # import the model and encoders
      X_train, y_train, encoder, lb = process_data(
         train, categorical_features=cat_features, label="salary", training=True
      )


    return results





