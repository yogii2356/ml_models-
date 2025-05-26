import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def data_load():
    np.random.seed(0)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = (X.ravel() > 5).astype(int) 
    return X,y

def model_train(X,y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

#  Save the model to a file
def savefile(model):
    joblib.dump(model,'logistic_regression_model.pkl')
    return True

def prediction(model):
    X_test = np.linspace(0, 10, 300).reshape(-1, 1)
    y_prob = model.predict_proba(X_test)[:, 1]
    return X_test,y_prob

def plot(x,y,X_test,y_prob):

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='darkorange', label='Actual Data')
    plt.plot(X_test, y_prob, color='royalblue', linewidth=2, label='Logistic Regression Curve')
    plt.xlabel('Independent Variable (X)', fontsize=12)
    plt.ylabel('Probability of Class 1 (y)', fontsize=12)
    plt.title('Logistic Regression Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def main():
    x,y = data_load()
    model = model_train(x,y)
    savefile(model)
    X_test,y_prob = prediction(model)
    plot(x,y,X_test,y_prob)
    
    
    
main()
    