#PS2 problem 1: effects of regularization

import numpy as np
import math
import matplotlib.pyplot as plt

'''
Error: errf = sum(i=1 to N)[ln(1+e**(-y_i*w.x_i))] + (lambda)*w.w
Gradient: graderrf[j] = sum(i=1 to N){[-y_i*x_ij*e**(-y_i*w.x_i)]/[1+e**(-y_i*w.x_i)]} + (2*lambda)*w_j
'''

def errf(w, x, y, lam): #error function for logistic error
    N = y.shape[0] #number of data points
    reg = lam * np.dot(w, w) #regularization term
    
    logsum = 0.
    for i in range(N):
        wx = np.dot(w, x[i])
        try:
            eywx = math.exp(-y[i]*wx)
            logsum += math.log1p(eywx)
        except OverflowError: # if exp(-ywx) is too high, make approximation log(1 + exp(-ywx)) ~ -ywx 
            logsum += -y[i]*wx
    
    return logsum + reg
    
def graderrf(w, x, y, lam):
    N = y.shape[0]
    M = w.shape[0] # should equal 14 since we have 13 features
    reg = (2*lam)*w
    expsum = np.array([0]*M, dtype=np.float64)
    
    for i in range(N):
        wx = np.dot(w, x[i])
        try:
            eywx = math.exp(-y[i]*wx)
            for j in range(M):
                expsum[j] -= (y[i]*x[i,j]*eywx)/(1+eywx)
        except OverflowError: # if exp(-ywx) is too high, make approximation (y[i]*x[i,j]*eywx)/(1+eywx) ~ y[i]*x[i,j]
            for j in range(M):
                expsum[j] -= y[i]*x[i,j]
            
    #print expsum
    #print reg
    
    return expsum + reg

choice = int(raw_input("Choose training set (1/2): "))
#plotchoice = raw_input("Choose plot option: a. Ein, b. Eout, c. |w| (a/b/c): ")
if choice == 1:
    training_file = 'wine_training1.txt'
elif choice == 2:
    training_file = 'wine_training2.txt'
else:
    print "Invalid choice!"
    exit(0)
    
#load training set
raw_data = np.loadtxt(training_file, delimiter=',')
#Nin = len(raw_data)
train_x = np.copy(raw_data)
train_x[:,0:1].fill(1) #first column is 1, other columns are the features
M = train_x.shape[1]
x_mean = np.array([1]*M, dtype=np.float64)
x_std = np.array([0]*M, dtype=np.float64)
train_x_norm = np.empty_like(train_x)
for i in range(1, M):
    x_mean[i] = np.mean(train_x[:,i])
    x_std[i] = np.std(train_x[:,i])
    train_x_norm[:,i] = (train_x[:,i] - x_mean[i])/ x_std[i]
train_x_norm[:,0:1].fill(1)
train_y = np.array(raw_data[:,0], dtype=int)

#load testing data
raw_data = np.loadtxt('wine_testing.txt', delimiter=',')
#Nout = len(raw_data)
test_x = np.copy(raw_data)
test_x[:,0:1].fill(1) #first column is 1, other columns are the features
test_x_norm = np.empty_like(test_x)
for i in range(1, M):
    test_x_norm[:,i] = (test_x[:,i] - x_mean[i])/ x_std[i]
test_x_norm[:,0:1].fill(1)
test_y = np.array(raw_data[:,0], dtype=int)

#initialize constants
epsilon = 0.00005
lambdalist = map(np.float64, raw_input("Input lambda parameters p. Lambda will be set to 1e-4 * 5^p: ").split())
lambdalist = map(lambda x: 0.0001*5**x, lambdalist)
eta = math.exp(-5)
maxiter = 1000

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(16, 4))

for lam in lambdalist:
    iterations = 0
    w = [np.array([0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], dtype=np.float64)]
    ein = [errf(w[-1], train_x_norm, train_y, 0)]
    eout = [errf(w[-1], test_x_norm, test_y, 0)]
    cont = True

    #do gradient descent
    while cont:
        #w -= eta*graderrf(w, train_x_norm,train_y, lam)
        w.append(w[-1] - eta*graderrf(w[-1], train_x_norm,train_y, lam))
        
        ein.append(errf(w[-1], train_x_norm, train_y, 0))
        eout.append(errf(w[-1], test_x_norm, test_y, 0))
        
        rel_loss_red = (ein[-1]-ein[-2])/(ein[1]-ein[0])
        if abs(rel_loss_red) <= epsilon:
            cont = False
            
        iterations += 1
        if iterations > maxiter:
            cont = False
    
    plotlabel = r'$\lambda = 10^{-4}\cdot5^%d$' % math.log(10**4*lam, 5)
    '''
    if plotchoice == 'a':
        plt.semilogy(range(len(ein)), ein, '-', label=plotlabel)
    elif plotchoice == 'b':
        plt.semilogy(range(len(eout)), eout, '-', label=plotlabel)
    elif plotchoice == 'c':
        wlength = len(w)
        wnorm = np.array([0]*wlength, dtype=np.float64)
        for i in range(wlength):
            wnorm[i] = np.linalg.norm(w[i])
        plt.semilogy(range(wlength), wnorm, '-', label=plotlabel)
    '''

    ax1.semilogy(range(len(ein)), ein, '-', label=plotlabel)
    ax2.semilogy(range(len(eout)), eout, '-', label=plotlabel)
    wlength = len(w)
    wnorm = np.array([0]*wlength, dtype=np.float64)
    for i in range(wlength):
        wnorm[i] = np.linalg.norm(w[i])
    ax3.semilogy(range(wlength), wnorm, '-', label=plotlabel)
    
    print "Lambda = %.4f" % lam
    print '-------------------'
    print "Iterations: %r" % iterations   
    print "Insample error is %.5f" % ein[-1]
    print "Outsample error is %.5f" % eout[-1]
    #print "Weights are %r" % w.tolist()
    print "Weight norm is %.5f" % np.linalg.norm(w[-1])
    print '--------------------------------------'

print '-------------------------------------------------------------------'
'''
plt.legend(loc='best')
if plotchoice == 'a':
    plt.title('Ein vs gradient step')
    plt.ylabel('Error')
elif plotchoice == 'b':
    plt.title('Eout vs gradient step')
    plt.ylabel('Error')
elif plotchoice == 'c':
    plt.title('|w| vs gradient step')
    plt.ylabel('Norm of weights')
plt.xlabel('Gradient Step');
'''
ax3.legend(loc='best', fontsize=10, ncol=2)
ax1.set_title(r'$E_{in}$ vs gradient step')
ax2.set_title(r'$E_{out}$ vs gradient step')
ax3.set_title(r'$|w|$ vs gradient step')
ax1.set_xlabel('Gradient Step')
ax2.set_xlabel('Gradient Step')
ax3.set_xlabel('Gradient Step')
ax1.set_ylabel('Error')
ax2.set_ylabel('Error')
ax3.set_ylabel('Norm of weights')
if choice == 1:
    ax1.set_ylim([0.5, 50])
    ax2.set_ylim([2, 20])
    ax3.set_ylim([0.1, 10])
elif choice == 2:
    ax1.set_ylim([0.2, 20])
    ax2.set_ylim([8, 24])
    ax3.set_ylim([0.2, 5])
plt.show()
