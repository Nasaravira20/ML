import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
import math

sns.set_theme()

data = pd.read_csv('stock_data.csv')
# print(data.columns)

# print(data.isnull().sum())
sns.pairplot(data)
# plt.show()

x = data[["Close","High","Low","Open","Volume"]].values
y = data[["Adj Close"]].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
model = LinearRegression()
model.fit(x_train,y_train)

# print(model.coef_)

# print(model.score(x_train,y_train))

predictions = model.predict(x_test)
print(r2_score(y_test,predictions))

dframe=pd.DataFrame({'actual':y_test.flatten(),'Predicted':predictions.flatten()}) 
print(dframe.head(15))

graph =dframe.head(10)
graph.plot(kind='bar')
plt.title('Actual vs Predicted')
plt.ylabel('Closing price')
# plt.show()

print('Mean Abs value:' ,metrics.mean_absolute_error(y_test,predictions))
print('Mean squared value:',metrics.mean_squared_error(y_test,predictions))
print('root mean squared error value:',math.sqrt(metrics.mean_squared_error(y_test,predictions)))
