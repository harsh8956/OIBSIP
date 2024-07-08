import pandas as pd

data=pd.read_csv('/kaggle/input/iriscsv/Iris.csv')
head=data.head()
print(head)

print(data.describe)
info=data.info()
print(info)

# check NULL value in data
print(data.isnull().sum())

column=data.columns
print(column)

y=data['Species']
x=data.drop(['Id','Species'],axis=1)

print(x)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

correlation=x.corr()
print(correlation)

plt.figure(figsize=(5,5))
sns.heatmap(correlation,annot=True)
plt.title("Heatmap for correlation")
plt.show()






from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.6,random_state=1554)

#model 1

# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()


#model 2

# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier()

# model 3

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()


predict=model.fit(x_train,y_train)

y_pred=model.predict(x_test)

#model accuracy ,performance

from sklearn.metrics import accuracy_score, confusion_matrix

conf_mat=confusion_matrix(y_pred,y_test)

print("confusion matrix = ")
print(conf_mat)

accuracy=accuracy_score(y_pred,y_test)
print(" accuracy = " ,accuracy*100,"%")

