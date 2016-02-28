from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel
import re
import numpy as np


HouseFile = open('HousingData.csv','r')
info_list = np.matrix([1,1,1,1])
house_list = []
price_matrix = np.matrix([1])
np.seterr(divide='ignore', invalid='ignore')

# Classes
class House(object):
    zip_code = 0
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

def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    mean_r = []
    std_r = []

    X_norm = X
    #np.insert(X_norm, 1, 1, axis=1)

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        #print(s)
        #print((X_norm[:, i] - m))
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r

def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta)

    sqErrors = (predictions.transpose() - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''

    (n,m) = y.shape
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)
        #print(predictions)
        #print(y)

        theta_size = theta.size

        for it in range(theta_size):
            temp = X[:, it]
            errors_x1 = (predictions - y) * temp
            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_cost(X, y, theta)
        

    return(theta, J_history)


#Load the dataset
for line in HouseFile:  
    matchObj = re.match( r'\d+ \w+ \w+,\w+,(\w+),\w+,(\d+),(\d+),(\d+),\w+,\w+ \w+ \d+ \d+:\d+:\d+ \w+ \d+,(\d+)', line, re.M|re.I)

    if matchObj:
        house = make_House(matchObj.group(1),matchObj.group(2),matchObj.group(3),matchObj.group(4),matchObj.group(5))
        new_value = np.matrix([float(matchObj.group(1)),float(matchObj.group(2)),float(matchObj.group(3)),float(matchObj.group(4))])
        info_list = np.concatenate((info_list, new_value), axis=0)
        new_price = np.matrix([float(matchObj.group(5))])
        price_matrix = np.concatenate((price_matrix, new_price), axis=0)
        house_list.append(house)

X = info_list.transpose()
y = price_matrix

#number of training samples
m = y.size

y.shape = (m, 1)

#Scale features and set them to zero mean
x, mean_r, std_r = feature_normalize(X)
#Add a column of ones to X (interception data)
it = ones(shape=(m, 4))
it[:, 0:4] = x.transpose()

#Some gradient descent settings
iterations = 100
alpha = 0.01

#Init Theta and Run Gradient Descent
theta = zeros(shape=(4, 1))

Theta, J_history = gradient_descent(it, y, theta, alpha, iterations)


