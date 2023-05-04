# Deploying ML Model To Cload Application Platform using FastAPI

In this Project, ML model using Random Forest built on the local system in order to predict salary classification of census data given 14 different features of population such as age,education,location and ...

After buiding the moodel, it is tested locally using pytest and deployed on Heroku platform using CI/CD (continous Integration and Continous Development) using Github actions workflow and Heroku Continous deployment feature. 

The model inference is used to predict salary of example data using FastAPI. 

requirements:

- FastAPI
- Pydantic
- Scikit-learn
- Numpy,Pandas
- requests
- pytest


### Heroku Link:

https://project-deployment-api.herokuapp.com/


### CI/CD:

CI: After pushing changes on the github, it automatically run pytest and check the test functions and if succesfull, it deploys on Heroku.
CD: Automatic deployment is enabled in Heroku GUI. 


### Model:

Random Forest classifier: Simple classifier with default parameters. 

### Metrics:
The model is evaluated using several metrics: Precsion,recall,fbeta

the results for test data which is 20% of the original data is as following:

- Precision: 72%
- Recall: 62%
- Fbeta: 67%

### Acknowledgement:

Thanks Udacity for this amazing project helping me develop my skills in MLops. This project is part of the MLops course practice project. 

### License:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:






