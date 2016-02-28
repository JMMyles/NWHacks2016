import numpy as np
from numpy import array
import re
import random

# Global variables 
HouseFile = open('HousingData.csv','r')
info_list = []
house_list = []
zip_array = []
beds_array = []
baths_array = []
sqr_feet_array = []

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

# Begin program
# Housing file string match
# 2400 INVERNESS DR,LINCOLN,95648,CA,3,2,1358,Residential,Thu May 15 00:00:00 EDT 2008,229027,38.897814,-121.324691
# Iterate through housefile for houseing information
for line in HouseFile:	
	matchObj = re.match( r'\d+ \w+ \w+,\w+,(\w+),\w+,(\d+),(\d+),(\d+),\w+,\w+ \w+ \d+ \d+:\d+:\d+ \w+ \d+,(\d+)', line, re.M|re.I)

	if matchObj:
		house = make_House(matchObj.group(1),matchObj.group(2),matchObj.group(3),matchObj.group(4),matchObj.group(5))
		info_list.append([matchObj.group(1),matchObj.group(2),matchObj.group(3),matchObj.group(4),matchObj.group(5)])
		zip_array.append([matchObj.group(1),matchObj.group(5)])
		beds_array.append([matchObj.group(2),matchObj.group(5)])
		baths_array.append([matchObj.group(3),matchObj.group(5)])
		sqr_feet_array.append([matchObj.group(4),matchObj.group(5)])
		house_list.append(house)

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
	for j in range(0, len(x)):
		for k in range(0, len(x[j])):
			x[j][k] = float(x[j][k])
	xTrans = zip(*x)
	for i in range(0, numIterations):
		hypothesis = np.dot(x, theta)
		loss = hypothesis - y
		# avg cost per example (the 2 in 2*m doesn't really matter here.
		# But to be consistent with the gradient, I include it)
		cost = np.sum(loss ** 2) / (2 * m)
		#print("Iteration %d | Cost: %f" % (i, cost))
		# avg gradient per example
		gradient = np.dot(xTrans, loss) / m
		# update
		theta = theta - alpha * gradient
	return theta

#beds
x_beds= beds_array[1:]
y_beds= beds_array[2:]
m_beds, n_beds = np.shape(beds_array)
#baths
x_baths= baths_array[1:]
y_baths= baths_array[2:]
m_baths, n_baths = np.shape(baths_array)
#sqr_feet
x_sqr_feet= sqr_feet_array[1:]
y_sqr_feet= sqr_feet_array[2:]
m_sqr_feet, n_sqr_feet = np.shape(sqr_feet_array)

numIterations= len(info_list)
alpha = 0.0005
theta_beds = np.ones(n_beds)
theta_baths = np.ones(n_beds)
theta_sqr_feet = np.ones(n_beds)
theta_beds = gradientDescent(x_beds, y_beds, theta_beds, alpha, m_beds, numIterations)
theta_baths = gradientDescent(x_baths, y_baths, theta_baths, alpha, m_baths, numIterations)
theta_beds = gradientDescent(x_beds, y_beds, theta_beds, alpha, m_beds, numIterations)
print(theta_beds)


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

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r


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



