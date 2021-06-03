import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris_class.csv')
# print(df.head())

df = df.drop(columns = ['Id'])
# print(df.head())

# print(df.describe())

# print(df.info())

# print(df.isnull().sum())

# print(df['SepalLengthCm'].hist())

le = LabelEncoder()

df['Species'] = le.fit_transform(df['Species'])
# print(df.head())

#train - 70
#test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

#logistic regression
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

reg.fit(x_train, y_train)

#performance

print("Accuracy:" ,reg.score(x_test,y_test)*100)

#knn
from sklearn.neighbors import KNeighborsClassifier
reg = KNeighborsClassifier()

reg.fit(x_train,y_train)

print("Accurracy: ",reg.score(x_test,y_test)*100)
 
pickle.dump(reg, open('specie.pkl','wb'))

model = pickle.load(open('specie.pkl','rb'))

print(model.predict([[5.1,3.5,1.4,0.2]]))