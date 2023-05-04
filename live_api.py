'''
live api check from heroku
'''
import requests
import json 



# request from the url and print results in console. 
data = {
        'age':  50,
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


# url of the app deployed on heroku
url = 'https://project-deployment-api.herokuapp.com/'

# path to make inference
req_path = url + 'predict'

# post url to make inference from the live app and get the result
req_inference = requests.post(req_path,data=json.dumps(data))

# check response status
status_code = req_inference.status_code
# check reponse result 
result = req_inference.json()['inference_result']

# print results on console and save screen shot on project directory
print(f"Status Code: {status_code}")
print(f"inference result: {result}")
