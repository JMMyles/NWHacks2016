from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel
import re
import numpy as np
import math
from sklearn import preprocessing, metrics, cross_validation
import urllib.request
import re
from yahoo_finance import Share
from sklearn.linear_model import SGDClassifier


HouseFile = open('HousingData.csv','r')
info_list = np.matrix([1,1,1])
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


NASDAQ = Share('^NQDXUSB')
Dow = Share('^DJI')
sp500 = Share('^GSPC')

def get_year_price(year, index):
    year = str(year-4)
    index_year = index.get_historical(year+'-01-01', year+'-12-31')
    s_open = float(index_year[0]['Open'])
    s_close = float(index_year[0]['Close'])
    s_open365 = float(index_year[len(index_year) - 1]['Open'])
    s_close365 = float(index_year[len(index_year) - 1]['Close'])
    s_open_diff = 1 + (s_open - s_open365) / s_open

    return(s_open_diff)

def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    mean_r = []
    std_r = []

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)

    return mean_r, std_r

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

    (n,m) = y.shape
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)
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
        if not(matchObj.group(1) == 0) or (matchObj.group(1) == 0) or (matchObj.group(3) == 0) or (matchObj.group(4) == 0)  or (matchObj.group(5) == 0): 
            house = make_House(matchObj.group(1),matchObj.group(2),matchObj.group(3),matchObj.group(4),matchObj.group(5))
            new_value = np.matrix([float(matchObj.group(2)),float(matchObj.group(3)),float(matchObj.group(4))])
            info_list = np.concatenate((info_list, new_value), axis=0)
            new_price = np.matrix([float(matchObj.group(5))])
            price_matrix = np.concatenate((price_matrix, new_price), axis=0)
            house_list.append(house)

X = info_list
y = price_matrix.transpose()
list_y = np.array(y)[0].tolist()

clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, list_y)
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)

def guess_Price(beds,baths,sqr_feet,year):
    if year == 2016:
        average = (Dow.get_change()+ sp500.get_change())/2
    else:
        average = (get_year_price(year,Dow) + get_year_price(year,sp500))/2
    print(average)
    print(Dow.get_change())
    return (clf.predict([[beds, baths,sqr_feet]]) / (average))

print(guess_Price(1,1,760,2009))

