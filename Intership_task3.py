from google.colab import files

uploaded = files.upload()

import io

import pandas as pd

data = pd.read_csv(io.BytesIO(uploaded['car data.csv']))

data.head()

data.info()

data.describe()

data.isnull().sum()

data.duplicated().sum()

data.drop_duplicates(inplace=True)

data.columns

data["Fuel_Type"].unique()

data["Selling_type"].unique()

data["Fuel_Type"].replace({'Petrol':0, 'Diesel':1, 'CNG':2},inplace=True)
data["Selling_type"].replace({'Dealer':0,'Individual':1},inplace=True)
data["Transmission"].replace({'Manual':0, 'Automatic':1},inplace=True)

data["Fuel_Type"].unique()

data["Transmission"].unique()

data.describe()

data.head()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

t=data.drop(['Car_Name'],axis=1)

correlation=t.corr()
print(correlation)

plt.figure(figsize=(15,8))
sns.heatmap(correlation,annot=True)
plt.title("Heatmap for Data")
plt.show()

y=data['Selling_Price']
x=data.drop(['Car_Name','Selling_Price','Selling_type'],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=50)


from sklearn.neighbors import KNeighborsRegressor

model=KNeighborsRegressor()

predict=model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(n_estimators=100, random_state=50)

predict=model.fit(x_train,y_train)
y_pred=model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')