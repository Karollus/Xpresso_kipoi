import joblib
import sys, os, h5py
import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn import preprocessing
from functools import reduce
from sklearn import linear_model
from scipy import stats

X_train = h5py.File('prepared_data/expr_preds.h5', 'r')['/train_in']
X_test = h5py.File('prepared_data/expr_preds.h5', 'r')['/test_in']
y_train = h5py.File('prepared_data/expr_preds.h5', 'r')['/train_out']
y_test = h5py.File('prepared_data/expr_preds.h5', 'r')['/test_out']
print(X_train.shape)
X_train = np.mean(X_train[:,499:500,:], axis=1) #promoter alone
X_test = np.mean(X_test[:,499:500,:], axis=1) #promoter alone
print(X_train.shape)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_hat = regr.predict(X_test)

for i in range(0,y_test.shape[1]):
     slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(y_test[:,i], y_hat[:,i])
     print('Test R^2 %d = %.3f' % (i, r_value**2))
