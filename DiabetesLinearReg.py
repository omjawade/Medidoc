import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from sklearn import svm
df = pd.read_csv("data/diabetes.csv")
#use required features
# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
regressordiabetes = svm.SVC(kernel='linear')

#Fitting model with trainig data
regressordiabetes.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressordiabetes, open('model.pkl','wb'))


#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5,166,72,19,175,25.8,0.587,51]]))
