import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

insurance_dataset = pd.read_csv('data/insurance.csv')
# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)
# encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
regressorinsurance =  RandomForestRegressor(n_estimators = 10, random_state = 0)
regressorinsurance.fit(X,Y)

print(regressorinsurance.predict([[19,1,27.9,0,0,1]]))
