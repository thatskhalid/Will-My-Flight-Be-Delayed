import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


df = pd.read_csv("/Users/khalidmahmood/Coding Workspace/Will-My-Flight-Be-Delayed/data/Airline_Delay_Cause.csv")
columns_im_dropping = ["carrier_name" , "airport_name" , "security_ct" , "security_delay"]
df.drop(columns = [x for x in columns_im_dropping if x in df.columns] , inplace=True)
#print(df.columns.tolist())
df.loc[:, "delay_rate"] = df["arr_del15"] / df["arr_flights"]
df.loc[:, "high_delay"] = (df["delay_rate"] > 0.2).astype(int)
df
df = df.dropna(subset= ["delay_rate"])

X = df.drop(columns = ["delay_rate"])
y = df["delay_rate"]

X = pd.get_dummies(X, columns = ["carrier", "airport"] , drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mse
X1 = df.drop(columns = ["high_delay"])
y1 = df["high_delay"]

X1 = pd.get_dummies(X1, columns = ["carrier", "airport"] , drop_first=True)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)

model1 = RandomForestClassifier()
model1.fit(X1_train, y1_train)

predictions1 = model1.predict(X1_test)

score = accuracy_score(y1_test, predictions1)
score
