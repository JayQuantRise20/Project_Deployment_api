from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds


def compute_metrics_slices(data,model,encoder,lb,process_data,category='education'):
    '''
    compute the model metrics on slices oof the data given category label
    '''
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
    unique_values = data[category].unique()

    with open('slice_output.txt', 'w') as f:
        for value in unique_values:
            sliced_data = data[data[category]==value]
            title = f" the sliced data on {category} with value {value} geenrates:\n"
            print(title)
            f.write(title)
            # compute the metrics for the category values
            # process sliced data
            X, y, _, _ = process_data(
                sliced_data, categorical_features=cat_features, label="salary", training=False,encoder=encoder, lb=lb
            )
            # predict sliced ys using model inference
            preds = inference(model,X)
            precision, recall, fbeta = compute_model_metrics(y,preds)

            f.write(f'Precision: {precision} , Recall: {recall} , Fbeta: {fbeta}\n')





    
