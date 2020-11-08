import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import turicreate


sales = pd.read_csv('home_data.csv')
sales.head()

# Find the zipcode with the largest average house price
zipcodes = sales['zipcode'].unique()

avg_price_zip=np.zeros((2,len(zipcodes)))
for i in range(len(zipcodes)):
	sales_zipcodes = sales[sales['zipcode']==zipcodes[i]]
	avg_price_zip[1,i] = np.mean(sales_zipcodes['price'])
	avg_price_zip[0,i] = zipcodes[i] 

print(np.sort(avg_price_zip,axis=1))

max_zip = '98199'


# Find house that have larger than 2000 sq ft and less than 4000 sq ft

filter_sales=sales[(sales['sqft_living']>=2000) & (sales['sqft_living']<=4000)]
print("Percentage of houses: ", len(filter_sales)/len(sales)*100)

# Building regression models - the first will use 5 filters, the second will use a lot more


# Define the feature sets
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]

sales = turicreate.SFrame('home_data.sframe/')
# Frist step - split the data set into training and test
traing_set,test_set = sales.random_split(0.8,seed=0)

# Second step - train models on the training set - it seems creating and training the model happens in the same call
simple_model = turicreate.linear_regression.create(traing_set,target="price",features=my_features,validation_set=None)
advanced_model = turicreate.linear_regression.create(traing_set,target="price",features=advanced_features,validation_set=None)

# Third Step - compute the rsme of the models using the test data sets
print("Simple Model:", simple_model.evaluate(test_set))
print("Advanced Model:", advanced_model.evaluate(test_set))

max_err_simple, rsme_simple = simple_model.evaluate(test_set)
max_err_advanced, rsme_advanced = advanced_model.evaluate(test_set)

 

