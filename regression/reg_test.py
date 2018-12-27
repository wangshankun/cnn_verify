#!/usr/bin/env python
# encoding: utf-8


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


X_train = np.random.random(size=[100,6])
Y_train = np.random.random(size=[100, 1])

model = linear_model.LinearRegression()
 
model.fit(X_train,Y_train)
 
a  = model.intercept_#截距
 
b = model.coef_#回归系数
 
print(a,b)

print(model.predict(np.random.random(size=[1,6])))