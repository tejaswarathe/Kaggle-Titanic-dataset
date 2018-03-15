# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 18:57:58 2018

@author: Tejas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Taking care of missing data for training set
#dropping rows from Embarked
train= train.dropna(subset=['Embarked'], how='any')
#dropping cabin column
train= train.drop('Cabin',axis=1)
#fill missing age values with mean
train['Age'].fillna(train['Age'].mean(), inplace=True)


# Taking care of missing data for test set
#dropping cabin column
test= test.drop('Cabin',axis=1)
#fill missing age values with mean
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)


#Defining test and train set
X_train = train.iloc[:, [2,4,5,6,7,9,10]].values
y_train = train.iloc[:, 1].values
X_test = test.iloc[:, [1, 3, 4, 5, 6, 8, 9]].values


# Encoding categorical data for training set
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_train_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_train_1.fit_transform(X_train[:, 1])
labelencoder_X_train_2 = LabelEncoder()
X_train[:, -1] = labelencoder_X_train_2.fit_transform(X_train[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

# Encoding categorical data for test set
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_test_1 = LabelEncoder()
X_test[:, 1] = labelencoder_X_test_1.fit_transform(X_test[:, 1])
labelencoder_X_test_2 = LabelEncoder()
X_test[:, -1] = labelencoder_X_test_2.fit_transform(X_test[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Exporting predictions as csv
prediction = pd.DataFrame(y_pred)
prediction.to_csv('Predictions.csv')