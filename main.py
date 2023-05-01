# Put the code for your API here.
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel
import json




app = FastAPI()

class IneferenceInput(BaseModel):
    age: int 
    workclass:str
    fnlgt:int
    education:str
    education-num:int
    marital-status:str
    occupation:str
    relationship:str
    race:str
    sex:str
    capital-gain:int
    capital-loss:int
    hours-per-week:int
    native-country:str
    
    

@app.get('/')
def greeting():

    return {'Greeting':'welcome to the model deploymnet using Fast API'}


@app.post('/predict/{item_id}')
async def predict(item_id:int, body: IneferenceInput):


     

    return 





