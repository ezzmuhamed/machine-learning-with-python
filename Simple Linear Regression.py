'''
Simple Linear Regression
About this Notebook
In this notebook, we learn how to use scikit-learn to implement simple linear regression.
We download a dataset that is related to fuel consumption and Carbon dioxide emission of cars. 
Then, we split our data into training and test sets,create a model using training set,
evaluate your model using test set, and finally use model to predict unknown value.
'''
#Importing Needed packages

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

#Downloading Data
#To download the data, we will use !wget to download it from IBM Object Storage

!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
#Reading the data in

df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

#Data Exploration

# summarize the data
df.describe()

#Lets select some features to explore more.


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#we can plot each of these features:


viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
#Now, lets plot each of these features vs the Emission, to see how linear is their relation:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()
'''
Creating train and test dataset
Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. 
After which, you train with the training set and test with the testing set.
This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have
been used to train the data. It is more realistic for real world problems.
This means that we know the outcome of each data point in this dataset, making it great to test with! 
And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. 
So, in essence, it is truly an out-of-sample testing.
Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. 
We create a mask to select random rows using np.random.rand() function:
'''
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
'''
Simple Regression Model
Linear Regression fits a linear model with coefficients  𝜃=(𝜃1,...,𝜃𝑛)  to minimize the 'residual sum of squares' between 
the independent x in the dataset, and the dependent y by the linear approximation.
'''
#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Modeling
#Using sklearn package to model data.
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

'''
Evaluation
we compare the actual values and predicted values to calculate the accuracy of a regression model.
Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:
Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand 
since it’s just average error.Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. 
It’s more popular than Mean absolute error because the focus is geared more towards large errors.
This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.
R-squared is not error, but is a popular metric for accuracy of your model.
It represents how close the data are to the fitted regression line.
The higher the R-squared, the better the model fits your data. 
Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
'''

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


