import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt

choice = raw_input('Lasso or ridge (l/r): ')

data = np.loadtxt('question2data.txt', delimiter=',')
X = data[:, 0:9]
y = data[:, 9]

if choice == 'l':
    plottitle = 'Weight decay with increasing lambda for lasso regularization'
    lambdalist = np.arange(0,2,0.01) #300 values
elif choice == 'r':
    plottitle = 'Weight decay with increasing lambda for ridge regularization'
    lambdalist = np.arange(0,15000,100) #300 values
else:
    exit(0)

lambdasize = lambdalist.size
W = np.empty((lambdasize,X.shape[1])) #300 rows, 9 columns. W[i] are the weights for lambda = lambdalist[i]

for i in range(lambdasize):
    if lambdalist[i] == 0:
        clf = lm.LinearRegression()
    elif choice == 'l':
        clf = lm.Lasso(alpha=lambdalist[i])
    elif choice == 'r':
        clf = lm.Ridge(alpha=lambdalist[i])
    else: 
        exit(0)
    clf.fit(X,y)
    W[i] = clf.coef_
    
for i in range(9):
    plotlabel = 'w_' + str(i+1)
    plt.plot(lambdalist, W[:,i], '-', label=plotlabel)

plt.legend(loc=1)
plt.title(plottitle)
plt.xlabel('Lambda')
plt.ylabel('Weights');
plt.show()