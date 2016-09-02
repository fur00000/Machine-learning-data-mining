#homework 1-4: Stochastic gradient descent

import random
import numpy as np
import math
import matplotlib.pyplot as plt

'''
Individual error: err = sum[(y-wx)**2]
Gradient: (d/dwn)err = -2*(y-wx)*x
'''

def errf(w, x, y): #x is Nxm dimensional matrix, y is N dimensional vector, w is m dimensional vector
    wx = np.dot(x,w)
    sqdif = (wx - train_y)**2
    return np.sum(sqdif)
  
def graderr(w,x,y): #w and x are m dimensional vectors, y is a scalar
    wx = np.inner(w,x)
    factor = 2*(wx-y)
    return factor*x

random.seed()

#load data
raw_data = []
file = open('hw1ds1.txt', 'r')
file.readline()
for line in file:
    x = line.split(',')
    raw_data.append(map(float,x))
file.close()
raw_data = np.array(raw_data)
train_x = np.ones((raw_data.shape[0],5))
train_x[:,1:] = raw_data[:,0:4]
train_y = raw_data[:,4]

#get closed form solution
xt = train_x.transpose()
xtx = np.dot(xt, train_x)
xtx_inv = np.linalg.inv(xtx)
xdag = np.dot(xtx_inv,xt)
w_closed = np.dot(xdag, train_y)

#Get closed form error
#train_gy = np.dot(train_x,w_closed)
#E_closed = (train_gy - train_y)**2
E_closed = errf(w_closed, train_x, train_y)
print "Closed-form solution error is %.5f" % np.sum(E_closed)
print "Closed-form weights are %r" % w_closed.tolist()

'''
#debug
print train_gy - train_y
print np.average(train_gy - train_y)
print np.max(train_gy - train_y)
print np.min(train_gy - train_y)
print np.std(train_gy - train_y)
print train_y.shape
'''

#initialize constants
epsilon = 0.0001
theta = map(float, raw_input("Input theta. Step size will be set to exp(-theta): ").split())
etalist = map(lambda x: math.exp(-x), theta)
epoch = int(raw_input("Input epoch size: "))
logplot = raw_input("Log plot? (y/n): ")

for eta in etalist:
    w = np.array([0, 0.001, 0.001, 0.001, 0.001], dtype=float)
    errors = [errf(w, train_x, train_y)]
    cont = True

    #do SGD
    while cont:
        N = train_y.size
        test_index = range(N)
        random.shuffle(test_index)
        wnext = w
        for j in test_index[0:epoch]:
            wnext = wnext - eta*graderr(wnext,train_x[j,:],train_y[j])
        
        w = wnext
        errors.append(errf(w, train_x, train_y))
        
        rel_loss_red = (errors[-1]-errors[-2])/(errors[1]-errors[0])
        if rel_loss_red <= epsilon:
            cont = False
    
    if logplot == 'y':
        plt.plot(range(len(errors)), map(math.log, errors), 'o-', label=math.log(eta))
    else:
        plt.plot(range(len(errors)), errors, 'o-', label=math.log(eta))

    epoch_numbers = len(errors)-1 
    '''    print epoch_numbers
        print errors[-1]-errors[-2]
        print errors[-1]-errors[-2]
        '''
       
    print "SGD number of epochs: %r" % epoch_numbers   
    print "SGD error is %.5f" % errors[-1]
    print "SGD weights are %r" % w.tolist()
    
plt.legend(loc='best')
plt.title('Error attenuation versus step size')
plt.xlabel('Epoch number')
if logplot == 'y':
    plt.ylabel('Log Error');
else:
    plt.ylabel('Error');
plt.show()