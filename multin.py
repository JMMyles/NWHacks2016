from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel
import re
import numpy as np


HouseFile = open('HousingData.csv','r')
info_list = np.matrix(["0","0","0","0"])
house_list = []
price_matrix = np.matrix(["0"])

# Classes
class House(object):
    zip_code = ""
    beds = 0
    baths = 0
    sqr_feet = 0
    price = 0

def make_House(zip_code, beds, baths,sqr_feet,price):
    house = House()
    house.zip_code = zip_code
    house.beds = beds
    house.baths = baths
    house.sqr_feet = sqr_feet
    house.price = price
    return house

def __repr__(self):
    return "<Test zip:%s beds:%s baths:%s sqr_feet:%s price:%s>" % (self.zip_code, self.beds,self.baths,self.sqr_feet,self.price)

def __str__(self):
    return "zip:%s beds:%s baths:%s sqr_feet:%s price:%s" % (self.zip_code, self.beds,self.baths,self.sqr_feet,self.price)

#Evaluate the linear regression


def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta)

    sqErrors = (predictions - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = (predictions - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history


#Load the dataset
for line in HouseFile:  
    matchObj = re.match( r'\d+ \w+ \w+,\w+,(\w+),\w+,(\d+),(\d+),(\d+),\w+,\w+ \w+ \d+ \d+:\d+:\d+ \w+ \d+,(\d+)', line, re.M|re.I)

    if matchObj:
        house = make_House(matchObj.group(1),matchObj.group(2),matchObj.group(3),matchObj.group(4),matchObj.group(5))
        new_value = np.matrix([matchObj.group(1),matchObj.group(2),matchObj.group(3),matchObj.group(4)])
        info_list = np.concatenate((info_list, new_value), axis=0)
        new_price = np.matrix([matchObj.group(5)])
        price_matrix = np.concatenate((price_matrix, new_price), axis=0)
        house_list.append(house)

X = info_list
y = price_matrix
num_iters = 100
alpha = 0.01
y_len = y.size
theta = np.zeros((y_len,1))
cost = compute_cost(X, y, theta)
'''
J = gradient_descent(X, y, theta, alpha, num_iters)
'''
'''
#number of training samples
m = y.size

y.shape = (m, 1)

#Scale features and set them to zero mean
x, mean_r, std_r = feature_normalize(X)

#Add a column of ones to X (interception data)
it = ones(shape=(m, 3))
it[:, 1:3] = x

#Some gradient descent settings
iterations = 100
alpha = 0.01

#Init Theta and Run Gradient Descent
theta = zeros(shape=(3, 1))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
print(theta)
print(J_history)
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

#Predict price of a 1650 sq-ft 3 br house
price = array([1.0,   ((1650.0 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house: %f' % (price))
'''
