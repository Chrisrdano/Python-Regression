"""
Python commands to build best fit linear quadratic models from input
data files.

Makes use of the Normal Equation.

Information on numpy matrices, functions, and attributes obtained from below:
https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matrix.html
"""
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

### Best Fit Line
dataset1 = pd.read_csv('xy_dataset.csv', names = ['x', 'y'])
##
# Define x and y 
x = dataset1.x
y = dataset1.y

# vector of ones
one_array = np.ones((1,100))
one_matrix = np.transpose(np.matrix(one_array))

### X and Y matrices

X = np.transpose(np.matrix(x))
Y = np.transpose(np.matrix(y))

### X with ones column

X = np.concatenate((X,one_matrix), axis = 1)

### Transpose X
XT = np.transpose(X)

#### XT*X
XX = np.matmul(XT, X)

#### XT*X inverse
inv_XX = np.linalg.inv(XX)


#### X * Y

XY = np.matmul(XT, Y)

#### weights (w coefficients)

w = np.matmul(inv_XX, XY)

w[0,0] # w1
w[1,0] # w0

Y_line = x*w[0,0] + w[1,0]
#######################################################################
#
# Best Fit Quadratic

x = dataset1.x
y = dataset1.y
x2 = x**2

## Ones vector
one_array = np.ones((1,100))
one_matrix = np.transpose(np.matrix(one_array))

### X and Y matrices
# X and Y row matrices
X2 = np.transpose(np.matrix(x2))

X = np.transpose(np.matrix(x))
Y = np.transpose(np.matrix(y))

### Combine 1, X, and X-squared

X = np.concatenate((one_matrix, X), axis = 1)
X = np.concatenate((X,X2), axis = 1)

# Normal Equation Computations
XT = np.transpose(X)
XX = np.matmul(XT, X)
inv_XX = np.linalg.inv(XX)
XY = np.matmul(XT, Y)

weights = np.matmul(inv_XX, XY)
weights
################################3
#### Calculate predicted values

Y_quad = (weights[2,0]*(x**2)) + (weights[1,0]*x) + weights[0,0]

########
# Sort data so the points are connected correctly
Y_quad_sorted = sorted(list(Y_quad))
x_sorted = sorted(list(x))
#################################
# Plotting

plot.title('Linear and Quadratic Regression Models')
plot.plot(x, y, 'o', color='black')
plot.plot(x, Y_line, color = 'red')
plot.plot(x_sorted, Y_quad_sorted, color = 'blue') 
plot.show()

# R-squared Values
R2 = 1 - (sum((y - Y_line)**2)/sum((y-np.mean(y))**2))
R2_line = R2
R2_line

R2 = 1 - (sum((y - Y_quad)**2)/sum((y-np.mean(y))**2))
R2_quad = R2
R2_quad
    
    