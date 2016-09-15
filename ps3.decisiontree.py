import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt

#load dataset
raw_data = np.loadtxt('wdbc.data', delimiter=',', dtype='string')
raw_y = raw_data[:,1]
N = len(raw_y) # number of data points, should = 569
y = np.array([0]*N, dtype=int)
for i in range(N):
    if raw_y[i] == 'M':
        y[i] = 1
    elif raw_y[i] == 'B':
        y[i] = -1
    else:
        print "Bad data point! Row %d" % i
        print raw_y[i]
        print raw_data[i]
        exit(0)        
x = raw_data[:,2:].astype(float)
D = x.shape[1] # number of features, should = 30

#separate into training and test set
train = 400
if N - train <= 0:
    print "Something went wrong!"
    exit(0)
train_x = x[:train]
train_y = y[:train]
test_x = x[train:]
test_y = y[train:]
Ntrain = len(train_y)
Ntest = len(test_y)

choice = raw_input('Vary minimum leaf node size (s), or maximum tree depth (d)? (s/d): ')
if choice == 's':
    param = 25
elif choice == 'd':
    param = 19
else:
    print "Bad input!"
    exit(0)

#learn
Ein = np.array([0.]*param, dtype=float) #in-sample error
Eout = np.array([0.]*param, dtype=float) #out-sample error
for i in range(param): 
    if choice == 's':
        mlns = i + 1 #minimum leaf node size from 1 to 25 inclusive
        clf = tree.DecisionTreeClassifier(criterion='gini',min_samples_split = mlns+1)
    elif choice == 'd':
        mtd = i + 2 #maximum tree depth size from 2 to 20 inclusive
        clf = tree.DecisionTreeClassifier(criterion='gini',max_depth = mtd)
    clf.fit(train_x, train_y)
    predict_train_y = clf.predict(train_x)
    predict_test_y = clf.predict(test_x)
    for j in range(Ntrain):
        if predict_train_y[j] != train_y[j]:
            Ein[i] += 1./Ntrain
    for j in range(Ntest):
        if predict_test_y[j] != test_y[j]:
            Eout[i] += 1./Ntest            
            
#plot
if choice == 's':
    plt.plot(range(1,26), Eout, 'o-', label=r'$E_{out}$')
    plt.plot(range(1,26), Ein, 'o-', label=r'$E_{in}$')
    plt.xlabel('Minimum leaf node size')
    plt.title('Error vs minimum leaf node size')
elif choice == 'd':
    plt.plot(range(2,21), Eout, 'o-', label=r'$E_{out}$')
    plt.plot(range(2,21), Ein, 'o-', label=r'$E_{in}$')
    plt.xlabel('Maximum tree depth')
    plt.title('Error vs maximum tree depth')
plt.legend(loc='best')
plt.ylabel('Error');
plt.show()