from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def performance_on_categorical_slice(
    data, feature, feature_value, categorical_features, label, encoder, lb, model
):
    """
    Computes the model metrics when a given categorical feature is held fixed.

    Inputs
    ------
    data : pd.DataFrame
        Full dataframe.
    feature : str
        The categorical feature to slice on.
    feature_value : str
        The value of the feature to hold fixed.
    categorical_features : list[str]
        List of categorical feature names.
    label : str
        Name of the label column.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer.
    model : sklearn model
        Trained model.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    from starter.ml.data import process_data

    slice_data = data[data[feature] == feature_value]
    if len(slice_data) == 0:
        return None, None, None

    X_slice, y_slice, _, _ = process_data(
        slice_data, categorical_features=categorical_features,
        label=label, training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
