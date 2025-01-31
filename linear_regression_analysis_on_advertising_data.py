# -*- coding: utf-8 -*-
"""Linear Regression Analysis on Advertising Data.ipynb

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/ML LAB/Advertising.csv")
print(data.head())

print(data.isnull().sum())

import plotly.express as px
import plotly.graph_objects as go
fig=px.scatter(data_frame=data,x="Sales",y="TV",size="TV", trendline="ols")
fig.show()

fig=px.scatter(data_frame=data,x="Sales",y="Newspaper",size="Newspaper", trendline="ols")
fig.show()

fig=px.scatter(data_frame=data,x="Sales",y="Radio",size="Radio", trendline="ols")
fig.show()

corr=data.corr()
print(corr["Sales"].sort_values(ascending=False))

x=np.array(data.drop(columns=["Sales"]))
y=np.array(data["Sales"])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))

