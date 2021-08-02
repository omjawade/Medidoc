import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("data/heart.csv")

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
regressorheart = LogisticRegression()

#Fitting model with trainig data
regressorheart.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressorheart, open('model.pkl','wb'))


#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[62,0,0,140,268,0,0,160,0,3.6,0,2,2]]))