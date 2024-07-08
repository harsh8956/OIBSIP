from google.colab import files

uploaded = files.upload()

import io

import pandas as pd

data = pd.read_csv(io.BytesIO(uploaded['Advertising.csv']))

data.info()

data.head()

data.describe()

data.columns

data.drop('Unnamed: 0',inplace=True,axis=1)

data.head()

data.columns

print(data.isnull().sum())

import seaborn as sns

sns.pairplot(data=data)

correlation=data.corr()

import matplotlib.pyplot as plt

sns.heatmap(correlation, annot=True)
plt.show()

y=data['Sales'];
x=data[['TV', 'Radio', 'Newspaper']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=40)

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')

from sklearn.neighbors import KNeighborsRegressor

model=KNeighborsRegressor()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')