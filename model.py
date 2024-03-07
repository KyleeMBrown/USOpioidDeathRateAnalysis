import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures 

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
pred_1_2021 = method1.predict([[2021]])
 # 5.73896104

#Write function to convert Death rates to Deaths
def conver(x):
    for i in x:
        return (x*(10**4))
    
#plot converted Deaths with reates
conver_pred_1 = conver(method1.predict(x_2d))

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

# Method 1 estimated that in 2021, the US Prescription Opiod Death rate will be 5.7 while the actual value is is 8.0   
    # score 0.75


#Method #2: Polynomial Regression

#Choose the degree of polynoial that best fits the shape of the data 
    # i'm focused on n = 2 and n = 3
plt.scatter(x,y, color = '#f0603c', label='Original Data Points')
plt.plot(x,y, '--', color='hotpink', label="shape")
plt.title('US Prescription Opium Death Rate')
plt.legend()

#Fit the  models, find coefficients and convert coefficients into a polynomial:
model1 = np.poly1d(np.polyfit(x,y, 1))
model2 = np.poly1d(np.polyfit(x,y, 2))
model3 = np.poly1d(np.polyfit(x,y, 3))
model4 = np.poly1d(np.polyfit(x,y, 4))

#Plot the models:
polyline = np.linspace(1999, 2021, 100)
plt.scatter(x,y, color = '#ff91fb', label ='Original Data Pts.')

plt.plot(polyline, model1(polyline), color='#ff8730', label='Model 1')
plt.plot(polyline, model2(polyline), color='#6c72c4', label='Model 2')
plt.plot(polyline, model3(polyline), color='#2c9a9c', label='Model 3')
plt.plot(polyline, model4(polyline), color='#d0db95', label='Model 4')
plt.title('Polynomial Regression of Data to n=4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

#Compare the models to actual data to find the best equation to use:
    #Changed my focus to n=3 and n=4 after graphing
polyline = np.linspace(1999, 2024, 100)


plt.subplot(2,2,1)
plt.scatter(x,y, color = '#ff91fb', label ='Original Data Pts.')
plt.plot(polyline, model1(polyline), color='#ff8730')
plt.plot(polyline, model2(polyline), color='#6c72c4')
plt.plot(polyline, model3(polyline), color='#2c9a9c')
plt.plot(polyline, model4(polyline), color='#d0db95')
plt.title('Polynomial Regression of Data to n=4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(2,2,3)
plt.scatter(x,y, color = '#ff91fb', label ='Original Data Pts.')
plt.plot(polyline, model3(polyline), color='#2c9a9c', label='Model 3')
plt.legend()

plt.subplot(2,2,4)
plt.scatter(x,y, color = '#ff91fb', label ='Original Data Pts.')
plt.plot(polyline, model4(polyline), color='#d0db95', label='Model 4')
plt.legend()

# model4 looks the most true to the data so I will predict the rate for 2021 using that model
pred_2_2021 = model4(2021)

plt.scatter(x,y, color = '#ff91fb', label ='Original Data Pts.')
plt.scatter(2021, 8.0, color="#a6f72d", label="Actual" )
plt.scatter(2021,pred_1_2021, color = '#d0db95', label ='Prediction #1(linear)')
plt.scatter(2021,pred_2_2021, color = '#95dbd4', label ='Prediction #2(poly)')
plt.plot(polyline, model4(polyline), color='#d0db95', label='Model 4')
plt.ylim(0, 8.5)
plt.legend()

#print polynomial to  fit model:
print(model4)

#method 3 splitting the data with polynomial regression:
X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=105, shuffle=True, test_size=0.33  )

# choose the polynomial features
poly = PolynomialFeatures(4)
# fit and transform training data
X_train_poly = poly.fit_transform(X_train)
# transform testing data
X_test_poly = poly.transform(X_test)

#use regression:
method3 = LinearRegression()
method3.fit(X_train_poly, y_train)

#get test and train predictions:
y_train_pred = method3.predict(X_train_poly)
y_test_pred = method3.predict(X_test_poly)

#plot train and test values
plt.subplot(2,2,1)
plt.scatter(X_train,y_train, color='#398aa3', label='Original Training Data Pts.')
x_values = np.linspace(X_train.max(), X_train.min(), 100)
x_values_poly = poly.transform(x_values)
y_pred_m3 =  method3.predict(x_values_poly)
plt.plot(x_values, y_pred_m3, color='orange', label="Train Set regression")
plt.legend()
plt.title('Polynomial regression Model of Opiod Death Rates using Split Data')

plt.subplot(2,2,4)
plt.scatter(X_test,y_test, color='#398aa3', label='Original Testing Data Pts.')
x_values = np.linspace(X_test.max(), X_test.min(), 100)
x_values_poly = poly.transform(x_values)
y_pred_m3 =  method3.predict(x_values_poly)
plt.plot(x_values, y_pred_m3, color ='hotpink', label='Test Set regression')
plt.xlim(1998, 2024)
plt.legend()

#Predict Death Rate in 2021 using Actual 2021
_2021_prediction3 = np.array([2021])
_2021_prediction3_new = poly.transform(_2021_prediction3.reshape(-1,1))
method3.predict(_2021_prediction3_new)

#plot train and test values
plt.subplot(1,2,1)
plt.scatter(X_train,y_train, color='#398aa3', label='Original Training Data Pts.')
x_values = np.linspace(X_train.max(), X_train.min(), 100)
x_values_poly = poly.transform(x_values)
y_pred_m3 =  method3.predict(x_values_poly)
plt.plot(x_values, y_pred_m3, color='orange', label="Train Set regression")
plt.scatter(2021, method3.predict(_2021_prediction3_new), color='#c5ede9', label='Predicted')
plt.scatter(2021, 8.0, color='#c4ffbd', label='Actual')
plt.legend(loc='lower right')
plt.xlim(1998, 2024)
plt.ylim(0, 11)
plt.title('Polynomial regression Model of Opiod Death Rates using Split Data')

plt.subplot(1,2,2)
plt.scatter(X_test,y_test, color='#398aa3', label='Original Testing Data Pts.')
x_values = np.linspace(X_test.max(), X_test.min(), 100)
x_values_poly = poly.transform(x_values)
y_pred_m3 =  method3.predict(x_values_poly)
plt.plot(x_values, y_pred_m3, color ='hotpink', label='Test Set regression')
plt.scatter(2021, method3.predict(_2021_prediction3_new), color='#c5ede9', label='Predicted')
plt.scatter(2021, 8.0, color='#c4ffbd', label='Actual')
plt.xlim(1998, 2024)
plt.ylim(0, 11)
plt.legend(loc='lower right')

#method 1 Prediction (Linear Regression):
# 5.73896104

#method 2 Prediction (Polynomial Regression):
# 4.842550277709961

#method 3 Prediction (Polynomial Regression Split Data):
# 4.37432097

#Find Average of all method predictions
Avg_pred_arr = np.array([5.73896104, 4.842550277709961, 4.37432097 ])
Avg_predict = Avg_pred_arr.mean() #4.985277429236653

#use model 1 to predict the nex 3 years after 2021:
predicted_vals = [method1.predict([[2022]]), method1.predict([[2023]]), method1.predict([[2024]])]
# [5.90677583]), array([6.07459063]), array([6.24240542]

#plot predicted values:
plt.scatter(x,y, color='orange', label="Original Data Pts.")
plt.scatter([2022,2023,2024], predicted_vals, color='hotpink', label = 'Model1 Predictions')
plt.title('Prediction of the Next 3 Years Using The most Accurate Model')
plt.legend(loc='lower right')