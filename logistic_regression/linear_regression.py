import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def load_salary_data():
    df = pd.read_csv("salary_data.csv")
    X = df[['YearsExperience']]
    y = df.iloc[:, -1]
    return X, y

def model_train(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_train, X_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return y_train_pred, y_test_pred

def evaluate(y_test, y_test_pred):
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    return mae, mse, rmse, r2

def main():
    X, y = load_salary_data()
    test_size = 0.2
    random_state = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Start MLflow tracking]
    mlflow.set_tracking_uri("http://192.168.1.100:5001")
    mlflow.set_experiment("Salary_Prediction_Experiment")

    with mlflow.start_run():
        model = model_train(X_train, y_train)
        y_train_pred, y_test_pred = predict(model, X_train, X_test)
        mae, mse, rmse, r2 = evaluate(y_test, y_test_pred)

        # Log parameters and metrics
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Print metrics
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

if __name__ == "__main__":
    main()
