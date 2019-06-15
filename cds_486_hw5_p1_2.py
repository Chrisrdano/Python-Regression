"""
Python Commands to contruct a best fit hyperplane model of six weights
"""
import numpy as np
import pandas as pd

data = pd.read_csv('xy_multidim_dataset.csv', 
                   names = ['x1', 'x2', 'x3', 'x4', 'x5','y'])

one_vector = np.ones((3500, 1))
one_matrix = np.matrix(one_vector)

x1m = np.transpose(np.matrix(data.x1))
x2m = np.transpose(np.matrix(data.x2))
x3m = np.transpose(np.matrix(data.x3))
x4m = np.transpose(np.matrix(data.x4))
x5m = np.transpose(np.matrix(data.x5))
Y = np.transpose(np.matrix(data.y))

X = np.concatenate((one_matrix, x1m, x2m, x3m, x4m, x5m), axis = 1)

# Normal Equation Computation
XT = np.transpose(X)
XX = np.matmul(XT, X)
inv_XX = np.linalg.inv(XX)
XY = np.matmul(XT, Y)

## Weights
w = np.matmul(inv_XX, XY)

# Predicted Y values
y_pred = w[0,0] + w[1,0]*data.x1 + w[2,0]*data.x2 + w[3,0]*data.x3 + w[4,0]*data.x4 + w[5,0]*data.x5

## R-squared Values
R2 = 1 - (sum((data.y - y_pred)**2)/sum((data.y-np.mean(data.y))**2))
print(R2)
