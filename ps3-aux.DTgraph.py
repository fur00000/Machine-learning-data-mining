import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydotplus as pydot

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
    
classes = ['Benign', 'Malignant']    
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension']
features *= 3
for i in range(len(features)):
    if 0<=i<10:
        features[i] = 'mean ' + features[i]
    elif 10<=i<20:
        features[i] = 'stderr ' + features[i]
    elif 20<=i<30:
        features[i] = 'worst ' + features[i]
    
choice = raw_input('Vary minimum leaf node size (s), or maximum tree depth (d)? (s/d): ')
if choice == 's':
    min_leaf_node_size = 13
    fname = 'ps3.min_leaf_node_size'
    
    #learn
    Ein = 0
    Eout = 0
    clf = tree.DecisionTreeClassifier(criterion='gini', 
        min_samples_split = min_leaf_node_size+1)
    clf.fit(train_x, train_y)
    predict_train_y = clf.predict(train_x)
    predict_test_y = clf.predict(test_x)
    for j in range(Ntrain):
        if predict_train_y[j] != train_y[j]:
            Ein += 1./Ntrain
    for j in range(Ntest):
        if predict_test_y[j] != test_y[j]:
            Eout += 1./Ntest
    
elif choice == 'd':
    max_tree_depth = 5
    fname = 'ps3.max_tree_depth'
    
    #learn
    Ein = 0
    Eout = 0
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth = max_tree_depth)
    clf.fit(train_x, train_y)
    predict_train_y = clf.predict(train_x)
    predict_test_y = clf.predict(test_x)
    for j in range(Ntrain):
        if predict_train_y[j] != train_y[j]:
            Ein += 1./Ntrain
    for j in range(Ntest):
        if predict_test_y[j] != test_y[j]:
            Eout += 1./Ntest
else:
    print "Bad input!"
    exit(0)    

print "In-sample error: %.5f" % Ein
print "Out-sample error: %.5f" % Eout    
    
#export tree
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data, feature_names = features, class_names = classes, filled=True, rounded=True) 
tree.export_graphviz(clf, out_file=fname+'.dot', feature_names = features, class_names = classes, filled=True, rounded=True) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf(fname+'.pdf')             
            