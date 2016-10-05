import numpy as np
from sklearn.preprocessing import Imputer

#load training data
raw_data = []
file = open('train.csv', 'r')
file.readline()
for line in file:
    raw_data.append(line.split(','))
file.close()
raw_data = np.array(raw_data)
y = map(int, raw_data[:,1])
feature_columns = {2: 'Pclass', 5: 'Sex', 6: 'Age', 7: 'SibSp', 8: 'Parch', 10: 'Fare', 12: 'Embarked'}
raw_X = raw_data[:,feature_columns.keys()]
X = np.empty(raw_X.shape)
for i in range(X.shape[0]):
    X[i,0] = int(raw_X[i,0]) #Pclass, or passenger class
    X[i,1] = 1 if raw_X[i,1]=='male' else -1 #Sex, 1 if male, -1 if female
    try:
        X[i,2] = float(raw_X[i,2]) #Age
    except ValueError:
        X[i,2] = np.nan #will replace later
    X[i,3] = int(raw_X[i,3]) #Sibsp, or number of siblings and spouses onboard
    X[i,4] = int(raw_X[i,4]) #Parch, or number of parents and children onbard
    X[i,5] = float(raw_X[i,5]) #Fare paid
    if raw_X[i,6][0] == 'C':
        X[i,6] = 0 #0 means from Cherbourg
    elif raw_X[i,6][0] == 'Q':
        X[i,6] = 1 #1 means from Queenstown
    elif raw_X[i,6][0] == 'S':
        X[i,6] = 2 #2 means from Southampton
    else:
        X[i,6] = np.nan
imp = Imputer(missing_values='NaN', strategy='median')
imp.fit(X)
X = imp.transform(X)

#load testing data
raw_data = []
file = open('test.csv', 'r')
file.readline()
for line in file:
    raw_data.append(line.split(','))
file.close()
raw_data = np.array(raw_data)
feature_columns = {1: 'Pclass', 4: 'Sex', 5: 'Age', 6: 'SibSp', 7: 'Parch', 9: 'Fare', 11: 'Embarked'}
raw_X = raw_data[:,feature_columns.keys()]
test_X = np.empty(raw_X.shape)
for i in range(test_X.shape[0]):
    test_X[i,0] = int(raw_X[i,0]) #Pclass, or passenger class
    test_X[i,1] = 1 if raw_X[i,1]=='male' else -1 #Sex, 1 if male, -1 if female
    try:
        test_X[i,2] = float(raw_X[i,2]) #Age
    except ValueError:
        test_X[i,2] = np.nan
    test_X[i,3] = int(raw_X[i,3]) #Sibsp, or number of siblings and spouses onboard
    test_X[i,4] = int(raw_X[i,4]) #Parch, or number of parents and children onbard
    try:
        test_X[i,5] = float(raw_X[i,5]) #Fare paid
    except ValueError:
        test_X[i,5] = np.nan
    if raw_X[i,6][0] == 'C':
        test_X[i,6] = 0 #0 means from Cherbourg
    elif raw_X[i,6][0] == 'Q':
        test_X[i,6] = 1 #1 means from Queenstown
    elif raw_X[i,6][0] == 'S':
        test_X[i,6] = 2 #2 means from Southampton
    else:
        test_X[i,6] = np.nan
test_X = imp.transform(test_X)

#Get method
print "Choose model. Options:"
print "1. Logistic regression"
print "2. Support vector classification (linear kernel)"
print "3. Support vector classification (RBF kernel)"
print "4. Decision trees"
print "5. Random forests"
print "6. Extra trees"
choice = int(raw_input("Choose model (1-6): "))

#Make prediction
if choice == 1:
    from sklearn import linear_model as lm
    clf = lm.LogisticRegressionCV()
    clf.fit(X, y)
    print "We chose from these C values for CV: ", 
    print clf.Cs_
    print "Best C value is: ", 
    print clf.C_
    print "Slope is: ", 
    print clf.coef_
    print "Intercept is: ", 
    print clf.intercept_
    howgood = clf.score(X, y)
    print "In-sample score is: %.5f" % howgood
    test_y = clf.predict(test_X)
    #print test_y
elif choice == 2:
    from sklearn import svm, grid_search
    svr = svm.SVC()
    param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}]
    clf = grid_search.GridSearchCV(svr, param_grid)
    clf.fit(X,y)
    howgood = clf.score(X, y)
    print "In-sample score is: %.5f" % howgood
    test_y = clf.predict(test_X)
elif choice == 3:
    from sklearn import svm, grid_search
    svr = svm.SVC()
    param_grid = [{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
    clf = grid_search.GridSearchCV(svr, param_grid)
    clf.fit(X,y)
    howgood = clf.score(X, y)
    print "In-sample score is: %.5f" % howgood
    test_y = clf.predict(test_X)
elif choice == 4: 
    from sklearn import tree
    from sklearn.externals.six import StringIO  
    import pydotplus as pydot
    md, ms = map(int, raw_input("Input max_depth, min_samples_split: ").split())
    clf = tree.DecisionTreeClassifier(criterion='gini',max_depth = md, min_samples_split = ms)
    clf.fit(X,y)
    howgood = clf.score(X, y)
    print "In-sample score is: %.5f" % howgood
    test_y = clf.predict(test_X)
    classes = ["Didn't survive", 'Survived']
    features = ['Pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'boarding']
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data, feature_names = features, class_names = classes, filled=True, rounded=True) 
    #tree.export_graphviz(clf, out_file='DT.dot', feature_names = features, class_names = classes, filled=True, rounded=True) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    filename = 'DT' + str(md) + '.pdf'
    graph.write_pdf(filename)   
elif choice == 5:
    from sklearn.ensemble import RandomForestClassifier
    rfparams = map(int, raw_input("Input n_estimators[, max_depth]: ").split())
    ne = rfparams[0]
    try:
        md = rfparams[1]
    except IndexError:
        md = None
    clf = RandomForestClassifier(n_estimators=ne, max_depth=md, min_samples_split=1)
    clf.fit(X,y)
    howgood = clf.score(X, y)
    print "In-sample score is: %.5f" % howgood
    test_y = clf.predict(test_X)
elif choice == 6:
    from sklearn.ensemble import ExtraTreesClassifier
    rfparams = map(int, raw_input("Input n_estimators[, max_depth]: ").split())
    ne = rfparams[0]
    try:
        md = rfparams[1]
    except IndexError:
        md = None
    clf = ExtraTreesClassifier(n_estimators=ne, max_depth=md, min_samples_split=1)
    clf.fit(X,y)
    howgood = clf.score(X, y)
    print "In-sample score is: %.5f" % howgood
    test_y = clf.predict(test_X)
else:
    print "Invalid input!"
    exit(0)    

#Write file
filename = 'answer' + str(choice) + '.csv'
file = open(filename, 'w')
file.write('PassengerID,Survived\n')
for i in range(len(test_y)):
    line = str(i+892) + ',' + str(test_y[i]) + '\n'
    file.write(line)
file.close()    