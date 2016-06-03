# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:56:18 2016

@author: vcarmean
"""

#DataSci Meetup Zillow Problem

import pandas as pd

df = pd.read_csv("zillow.csv", parse_dates=['date_sold'])
[ x for x in df.columns]

df.dropna(inplace=True)
df.sort_values(by="date_sold", inplace=True)

price = df[['date_sold', 'price']]

pd.rolling_mean(price.groupby('date_sold').agg('mean'), window=30).dropna().plot()

# models that may be useful
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn import cross_validation