

#Import scikit-learn dataset library
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

#Load dataset
cancer = datasets.load_breast_cancer()
print(cancer)

print(cancer.data.shape)
print(cancer.target.shape)
print(cancer.target_names)
print(cancer.feature_names)

#import the necessary module
from sklearn.model_selection import train_test_split

X=cancer.data
Y=cancer.target
#split data set into train and test sets
X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.50, random_state = 100)

#Create a Decision Tree Classifier (using Gini)
clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=7, min_samples_leaf=10)

clf_gini.fit(X_train, y_train)

# Predict the classes of test data
y_pred = clf_gini.predict(X_test)
print("Predicted values:")
print(y_pred)

# Model Accuracy
from sklearn import metrics
print("Confusion Matrix: ",
        metrics.confusion_matrix(y_test, y_pred))
print ("Accuracy : ",
    metrics.accuracy_score(y_test,y_pred)*100)
print("Report : ",
    metrics.classification_report(y_test, y_pred))