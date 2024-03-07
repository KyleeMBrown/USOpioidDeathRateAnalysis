import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression

pd.options.display.max_columns = 999
data = pd.read_csv('drug-overdose-death-rates new.csv')
d= data.drop(['Entity', 'Code', 'Any opioid death rates (CDC WONDER)',
       'Cocaine overdose death rates (CDC WONDER)',
       'Heroin overdose death rates (CDC WONDER)',
       'Synthetic opioids death rates (CDC WONDER)'], axis=1)

x = d['Year']
y = d['Prescription Opioids death rates (US CDC WONDER)']

#plot 1

plt.subplot(1,3,1)
plt.bar(x,y, color="hotpink")
plt.xlabel('Year')
plt.ylabel('Opiod Death Rates')

#plot 2

plt.subplot(1,3,3)
plt.scatter(x,y, color="#99FFCC")
plt.xlabel('Year')
plt.ylabel('Opiod Death Rates')

#Compare plots with corr()
d.corr()

#plot: 
plt.scatter(x,y, color="#99FFCC")
plt.plot(x,y, '--', color='hotpink')
plt.xlabel('Year')
plt.ylabel('Opiod Death Rates')

x_2d = d[['Year']]

#Goal one: 
#   find rate of Change at which death rates increase from 1999-2020

#Method 1: Perform Linear Regression to predict the 2021 Death Rate

method1 = LinearRegression()
method1.fit(x,y)
#Use found polynomial equation to predict y:
method1.predict(x)

#Find the score of the model
method1.score(x,y)

#plot values:

y_pred = method1.predict(x)
plt.scatter(x,y, color="blue", label ="Data Points")
plt.plot(x, y_pred, '--', color="hotpink", label="Linear Regression")
plt.title('Linear Regression of Opiod Death Rate')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

#Predict opiod death rate in 2021:
method1.predict([[2021]]) # 5.73896104

#Write function to convert Death rates to Deaths
def conver(x):
    for i in x:
        return (x*(10**4))
    
#plot converted Deaths with reates
conver_pred_1 = conver(method1.predict(x_2d))
conver_pred_1

plt.subplot(1, 2, 1)
plt.scatter(x,conver(y), color="orange", label ="Data Points(#Deaths)")
plt.plot(x, conver_pred_1, '--', color="red", label="Linear Regression")
plt.title('Linear Regression of Opiod Death Rate')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(1, 2, 2)
y_pred = method1.predict(x_2d)
plt.scatter(x,y, color="blue", label ="Data Points(Rate)")
plt.plot(x, y_pred, '--', color="hotpink", label="Linear Regression")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

