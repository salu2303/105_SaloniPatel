

#Import library
from sklearn import preprocessing 
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

#Predictor variables
Outlook = ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast',
            'Rainy', 'Rainy', 'Sunny', 'Rainy','Overcast', 'Overcast', 'Sunny']
Temperature = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
Humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
            'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
Wind = ['False', 'True', 'False', 'False', 'False', 'True', 'True',
            'False', 'False', 'False', 'True', 'True', 'False', 'True']

#Class Label:
Play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

#creating OneHotEncoder
enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
weather=pd.DataFrame(Outlook, columns=['Outlook'])
weather['Temperature']=pd.DataFrame(Temperature, columns=['Temperature'])
weather['Humidity']=pd.DataFrame(Humidity, columns=['Humidity'])
weather['Wind']=pd.DataFrame(Wind, columns=['Wind'])
dum_df = pd.get_dummies(weather, columns=["Outlook","Temperature","Humidity","Wind"])
weather = weather.join(dum_df)
Play_encoded = preprocessing.LabelEncoder().fit_transform(Play)

from sklearn.model_selection import train_test_split
X=weather.values[:,5:15]
Y=Play_encoded
X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.15, random_state = 100)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

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