# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing...
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing, cross_validation, svm
import seaborn as sns


import datetime,time
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')



#IMPORTING THE DATASET
df=pd.read_csv('g:/data/TC1-HDFCBANK.csv')


#TO PRINT THE DATA SET USE BELOW LINE
#print(df.head())

#TAKING EACH ATTRIBUTE
x1=df['Open Price']
x2=df['High Price']
x3=df['Low Price']
x4=df['Last Traded Price']
x5=df['Close Price']
x6=df['Total Traded Quantity']
y=df['Turnover (in Lakhs)']


#NOW TAKING EACH ATTRIBUTE AND COMBINING INTO ONE LARGE ARRAY
l=[x1,x2,x3,x4,x5,x6]

#OR WE CAN DIRECTLY USE THIS BUT UNDERSTANDING WE ARE DOING ABOVE METHOD 
u=['Open Price','High Price','Low Price','Last Traded Price','Close Price','Total Traded Quantity']

#FOR PLOTTING GRAPH
"""
#plotting x1 vs x2
plt.scatter(x1,x2)
plt.xlabel('open price')
plt.ylabel('high price')
plt.show()
"""

#graph between x1 to x6
for i in range(0,6):
    print('for variable:',u[i])
    for j in range(0,6):
        if(i!=j):
            plt.scatter(l[i],l[j])
            plt.xlabel(u[i])
            plt.ylabel(u[j])
            plt.show()

 
#plotting X attributes to Y
plt.scatter(x1,y)
plt.xlabel('open price')
plt.ylabel('Turnover (in Lakhs)')
plt.show()

plt.scatter(x2,y)
plt.xlabel('High Price')
plt.ylabel('Turnover (in Lakhs)')
plt.show()

plt.scatter(x3,y)
plt.xlabel('Low Price')
plt.ylabel('Turnover (in Lakhs)')
plt.show()

plt.scatter(x4,y)
plt.xlabel('Last Traded Price')
plt.ylabel('Turnover (in Lakhs)')
plt.show()

plt.scatter(x5,y)
plt.xlabel('Close Price')
plt.ylabel('Turnover (in Lakhs)')
plt.show()

plt.scatter(x6,y)
plt.xlabel('Total Traded Quantity')
plt.ylabel('Turnover (in Lakhs)')
plt.show()


#converting each attibute variable into more useful formate that is numpy
x1=np.array(x1)
x1.reshape(len(x1),1)
x2=np.array(x2)
x2.reshape(len(x2),1)
x3=np.array(x3)
x3.reshape(len(x3),1)

x4=np.array(x4)
x4.reshape(len(x4),1)
x5=np.array(x5)
x5.reshape(len(x5),1)
x6=np.array(x6)
x6.reshape(len(x6),1)


df = df[['Open Price','High Price','Low Price','Last Traded Price','Close Price','Total Traded Quantity','Turnover (in Lakhs)']]
#ignore next line
#dF=df[['Open Price','Total Traded Quantity','Turnover (in Lakhs)']]    


#print(df.head())


#extracting tables and dividing into to using variable X, prediction variable Y.


# df.drop(columns=['B', 'C']) or df.drop(['B', 'C'], axis=1) it drop column B and C
#df.drop([0, 1]) this will drop specified rows. i.e 1 and 0 row droped.
#X = np.array(df.drop(['label'], 1)) here 1 stands for column and label is dropeed from df and put into x.
#y = np.array(df['label']) here only label is inserted.
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values


#Generally, you want your features in machine learning to be in a range of -1 to 1. 
#This may do nothing, but it usually speeds up processing and can also help with accuracy
X = preprocessing.scale(X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

#accuracy libaility or confidence score are samething
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#for forecssating data for next 30 days.
forecast_out = 30
X_lately = X[1:forecast_out]
X = X[forecast_out:]
df.dropna(inplace=True)



"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""


#classifier support vector machine.
clf1 = svm.SVR()
clf1.fit(X_train, y_train)
confidence = clf1.score(X_test, y_test)
print('\n\nsvm:')
print(confidence)
#forecasting data
forecast_set = clf1.predict(X_lately)
print('forecasting done through SVM:\n',forecast_set,'\naccuracy %:' ,confidence*100)



#classifier linear regression
clf2 = LinearRegression()
clf2.fit(X_train, y_train)
confidence = clf2.score(X_test, y_test)
print('\n\n\nlinear regression:')
print(confidence)
#forecasting data
forecast_set = clf2.predict(X_lately)
print('forecasting done through linear regression:\n',forecast_set,'\naccuracy %:' ,confidence*100)



 
#using polynomial
clf3 = svm.SVR(kernel='poly') 
clf3.fit(X_train, y_train)
confidence = clf3.score(X_test, y_test)
print('\n\npoly:\n',confidence)
#forecasting data
forecast_set = clf3.predict(X_lately)
print('forecasting done through polynomial:','\n',forecast_set,'\naccuracy %:' ,confidence*100)




#using rbf classifier
clf4 = svm.SVR(kernel='rbf') 
clf4.fit(X_train, y_train)
confidence = clf4.score(X_test, y_test)
print('\n\n\nrbf:\n',confidence)
#forecasting data
forecast_set = clf4.predict(X_lately)
print('forecasting done through RBF:\n',forecast_set,'\naccuracy %:' ,confidence*100)


#using sigmoidal function
clf5 = svm.SVR(kernel='sigmoid') 
clf5.fit(X_train, y_train)
confidence = clf5.score(X_test, y_test)
print('\n\n\nsigmoid\n',confidence)
#forecasting data
forecast_set = clf5.predict(X_lately)
print('forecasting done through sigmoidal function:\n',forecast_set,'\naccuracy %:' ,confidence*100)
   


