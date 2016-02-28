import numpy as np
import re
import random

# Global variables 
HouseFile = open('HousingData.csv','r')
info_list = []
house_list = []
zip_array = []
beds_array = []
bath_array = []
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
		beds_array.append([matchObj.group(1),matchObj.group(5)])
		bath_array.append([matchObj.group(1),matchObj.group(5)])
		sqr_feet_array.append([matchObj.group(1),matchObj.group(5)])
		house_list.append(house)

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
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


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise

#beds
x_zip, y_zip = zip_array
m_zip, n_zip = np.shape(x_zip)
#baths
x_zip, y_zip = zip_array
m_zip, n_zip = np.shape(x_zip)
#sqr_feet
x_zip, y_zip = zip_array
m_zip, n_zip = np.shape(x_zip)

numIterations= len(info_list)
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x_zip, y_zip, theta, alpha, m_zip, numIterations)
print(theta)
print(x.shape)




