# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the dataset containing input features and binary class labels.
2. Initialize the model parameters (weights and bias), learning rate, and the number of iterations.
3. Apply the Gradient Descent algorithm by computing the sigmoid function, calculating gradients, and updating the parameters iteratively.
4. Predict the class labels using the trained model, evaluate the accuracy, and display the results.

## Program:

~~~py

import numpy as np

# Sample Dataset
# Features: [Hours Studied, Attendance]
X = np.array([
    [2, 40],
    [3, 45],
    [4, 50],
    [5, 55],
    [6, 60],
    [7, 65],
    [8, 70],
    [9, 75],
    [10, 80],
    [11, 85]
], dtype=float)

# Target (0 = Fail, 1 = Pass)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Feature Scaling
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Initialize Parameters
m, n = X.shape

weights = np.zeros(n)

bias = 0

learning_rate = 0.1

iterations = 1000

# Sigmoid Function
def sigmoid(z):

    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(iterations):

    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)

    dw = (1 / m) * np.dot(X.T, (predictions - y))
    db = (1 / m) * np.sum(predictions - y)

    weights -= learning_rate * dw
    bias -= learning_rate * db

# Predictions
probabilities = sigmoid(np.dot(X, weights) + bias)

y_pred = (probabilities >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_pred == y) * 100

print("Weights:", np.round(weights, 4))

print("Bias:", round(bias, 4))

print("\nPredicted Labels:")

print(y_pred)

print("\nActual Labels:")

print(y)

print("\nAccuracy: {:.2f}%".format(accuracy))

# Prediction for a new student
new_data = np.array([[7, 68]], dtype=float)

new_data = (new_data - X.mean(axis=0)) / X.std(axis=0)

prob = sigmoid(np.dot(new_data, weights) + bias)

if prob >= 0.5:

    print("\nPrediction: Pass")
else:

    print("\nPrediction: Fail")

~~~

## Output:

<img width="797" height="601" alt="image" src="https://github.com/user-attachments/assets/b2a9a814-19b1-4692-954b-fb8f3f0a96b8" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

