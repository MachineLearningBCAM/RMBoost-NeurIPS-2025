#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on
"""

import numpy as np

import scipy.io as sio
from scipy import stats

from RMBoost import fit, predict_boost

#  Load data
data = sio.loadmat('../../data/diabetes.mat')

x=data['X']
y=data['y']

x = stats.zscore(x, axis = 1)

T = 200 # Number of rounds
solver = "linprog" # linprog or mosek

x_train = x
y_train = np.squeeze(y)
    

n_train = len(y_train)

#  Fit the model
model, upper = fit(x_train, y_train, T, solver, n_samples=1000)

# Predict
y_pred = predict_boost(model, x_train)

# Error
error_train=np.sum(y_pred!=y_train)/n_train

print('The training error is ', error_train)
