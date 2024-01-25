import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from trading_signal import compute_trading_signal
from sklearn.model_selection import GridSearchCV
import numpy as np

df = pd.read_csv('../data/BELEX15_.csv')

def LS_SVM_(data_source:str):
    
    np.random.seed(42)

    df = pd.read_csv(data_source)

    # Create a y column for the training part
    df['y'] = df['price'].pct_change().shift(-1) > 0

    # Shift the feature columns to use t-1 metrics
    df[['MACD', 'EMA_10', 'Log_Return']] = df[['MACD', 'EMA_10', 'Log_Return']].shift(1)

    # Drop the first row as it will have NaN values after shifting
    df = df.dropna()

    # Define the date ranges for train and test
    train_start_date = '2010-01-01'
    train_end_date = '2017-12-31'
    test_start_date = '2018-01-01'
    test_end_date = '2022-12-31'

    # Filter the DataFrame based on the date ranges
    train_data = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)]
    test_data = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]

    # reset the index of the resulting DataFrames
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    # Select relevant features
    X_train = train_data[['MACD', 'EMA_10', 'Log_Return']]
    y_train = train_data['y']
    X_test = test_data[['MACD', 'EMA_10', 'Log_Return']]
    y_test = test_data['y']

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LS-SVM classifier with 'rbf' kernel
    svm = SVC(kernel='rbf',random_state=42)

    #Define the parameter grid for grid search
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}

    # Create a GridSearchCV object with 10-fold cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=10)

    # Fit the model to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Access the best trained LS-SVM classifier
    best_svm = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_svm.predict(X_test_scaled)
    
    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    #compute trend predictions
    test_data['y'] = y_pred
    
    # Concatenate the train and test datasets back together
    df = pd.concat([train_data, test_data], ignore_index=True)
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Shift the feature columns back
    df[['MACD', 'EMA_10', 'Log_Return']] = df[['MACD', 'EMA_10', 'Log_Return']].shift(-1)
    
    compute_trading_signal(df)
    
    return df