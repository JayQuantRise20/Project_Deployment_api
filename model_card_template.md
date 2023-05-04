# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is simple RrandomForestClassifier trained on census data. It is not fine tuned and the parameters are the defualt parameters,but 
updating the parameters can increase  perofmance of the model. 

## Intended Use
It is used to predict the salary of the population and categorize them based 
on training data and labels for salary categories. 
## Training Data
the data is split on training and test with 80% and 20% of the original data. It contains categorical and numerical columns 
with categories encoded using OneHotEncoder object in scikit-learn. Overall categorical features  are 8 features. 

## Evaluation Data
Evaluation data is 20% of the original data. 

## Metrics
The model is evaluated based on precision,recall and fbeta scores using scikit-learn metrics. 

the results for test data:

- precision: 0.729829
- recall: 0.626031
- fbeta score : 0.673957

## Ethical Considerations
This data is population data containing private data of the individuals. There is a risk of exposing data 
on public. So, it shoulld be considered that the data should not be used in public without the consent of the 
population, or take further actions to do not reveal any personal inforamtoion of the individuals

## Caveats and Recommendations
The model can be improved by fine tuning the hyper parameters of the random forest classifier. Also,
new algorithms can be used to predict the salary such as regression models. 