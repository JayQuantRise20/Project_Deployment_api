# Script to train machine learning model.
from joblib import dump,load
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import train_model,compute_metrics_slices
from ml.data import process_data


# Add code to load in the data.
data = pd.read_csv("./data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20) 

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test,_,_,_ = process_data(test,categorical_features=cat_features,label = 'salary',training=False,encoder=encoder, lb=lb)
# Train and save a model.
model = train_model(X_train,y_train)

# evalue the model on data slices (here education) and print in slice_output.txt
encoder = encoder = load('model/encoder.joblib')
compute_metrics_slices(data,model,encoder,lb,process_data)

# save the model
dump(model,'./model/random_forest.joblib')
# save the encoder
dump(encoder,'./model/encoder.joblib')
# save the labelizer
dump(lb,'./model/lb.joblib')


