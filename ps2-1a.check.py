#PS2 problem 1 check it with sklearn

import numpy as np
import math
from sklearn import linear_model as lm
#import matplotlib.pyplot as plt

'''
Error: errf = sum(i=1 to N)[ln(1+e**(-y_i*w.x_i))] + (lamda/N)*w.w
Gradient: graderrf[j] = sum(i=1 to N){[-y_i*x_ij*e**(-y_i*w.x_i)]/[1+e**(-y_i*w.x_i)]} + (2*lambda/N)*w_j
'''

def errf(w, x, y, lam): #error function for logistic error
    N = y.shape[0] #number of data points
    reg = lam * np.dot(w, w) / float(N) #regularization term
    
    logsum = 0.
    for i in range(N):
        wx = np.dot(w, x[i])
        eywx = math.exp(-y[i]*wx)
        logsum += math.log(1 + eywx)
    
    return logsum + reg
    
def errfa(wx, y): #error function for logistic error
    N = y.shape[0] #number of data points
    
    logsum = 0.
    for i in range(N):
        eywx = math.exp(-y[i]*wx[i])
        logsum += math.log(1 + eywx)
    
    return logsum    
    
def graderrf(w, x, y, lam):
    N = y.shape[0]
    M = w.shape[0] # should equal 14 since we have 13 features
    reg = (2*lam/float(N))*w
    expsum = np.array([0]*M, dtype=np.float64)
    
    for i in range(N):
        wx = np.dot(w, x[i])
        eywx = math.exp(-y[i]*wx)
        for j in range(M):
            expsum[j] -= (y[i]*x[i,j]*eywx)/(1+eywx)
    
    return expsum + reg

choice = int(raw_input("Choose training set (1/2): "))
if choice == 1:
    training_file = 'wine_training1.txt'
elif choice == 2:
    training_file = 'wine_training2.txt'
else:
    print "Invalid choice!"
    exit(0)
    
#load training set
raw_data = np.loadtxt(training_file, delimiter=',')
Nin = len(raw_data)
train_x = np.copy(raw_data)
train_x[:,0:1].fill(1) #first column is 1, other columns are the features
train_y = np.array(raw_data[:,0], dtype=int)

#load testing data
raw_data = np.loadtxt('wine_testing.txt', delimiter=',')
Nout = len(raw_data)
test_x = np.copy(raw_data)
test_x[:,0:1].fill(1) #first column is 1, other columns are the features
test_y = np.array(raw_data[:,0], dtype=int)

#initialize constants
epsilon = 0.00005
lambdalist = map(np.float64, raw_input("Input lambda values: ").split())
eta = math.exp(-16)

for lam in lambdalist:
    ein=[]
    eout=[]
    #clf = lm.Ridge(alpha=lam)
    if lam == 0:
        clf = lm.LogisticRegression()
    else:
        clf = lm.LogisticRegression(penalty='l2', C=1/lam)
    clf.fit(train_x[:,1:],train_y)
    w = np.array([0]*14, dtype=np.float64)
    w[0] = clf.intercept_
    w[1:] = clf.coef_
    train_xw = np.dot(train_x,w)
    test_xw = np.dot(test_x,w)
    ein.append(errfa(train_xw, train_y))
    eout.append(errfa(test_xw, test_y))
    #ein.append(errf(w, train_x, train_y, lam))
    #eout.append(errf(w, test_x, test_y, lam))
    #gy = clf.predict(train_x[:,1:])
    #hy = clf.predict(test_x[:,1:])
    #ein.append(errfa(gy, train_y))
    #eout.append(errfa(hy, test_y))
     
    print "Insample error is %.5f" % ein[-1]
    print "Outsample error is %.5f" % eout[-1]
    print "Weights are %r" % w.tolist()
    print '--------------------------------------'
