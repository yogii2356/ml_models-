from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score

# lode iris_data set 
def lode_data():
    iris = load_iris()
    numSamples, numFeatures = iris.data.shape
    print(numSamples, numFeatures)
    print(list(iris.target_names))
    X = iris.data
    y = iris.target
    return X,y,iris

# Scatter plot: Petal length vs Petal width

def plot(X,y,iris):
    plt.figure(figsize=(8, 6))
    for i, species in enumerate(iris.target_names):
        plt.scatter(X[y == i, 2], X[y == i, 3], label=species)
    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])
    plt.title("Iris Dataset - Petal Length vs Width")
    plt.legend()
    plt.grid(True)
    plt.show()

# Initialize and train the XGBoost classifier
def model_train(X_train, y_train):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    return model
    

# Save the model to a file
def save_model(model):
    joblib.dump(model, 'iris_xgboost_classification_model.pkl')
 
def prediction(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred


#  Evaluate the model's accuracy
def evalution(model,y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return accuracy
    
def main():
    x,y,iris = lode_data()
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    model = model_train(X_train, y_train)
    save_model(model)
    y_pred = prediction(model,X_test)
    accuracy = evalution(model,y_test, y_pred)
    plot(x,y,iris)
    
    
main()

