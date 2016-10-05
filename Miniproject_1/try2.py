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
feature_columns = {2: 'Pclass', 4: 'Title', 5: 'Sex', 6: 'Age', 7: 'SibSp', 8: 'Parch', 9: 'Ticket', 10: 'Fare', 12: 'Embarked'}
raw_X = raw_data[:,feature_columns.keys()]
for i in range(raw_X.shape[0]):
    # get the title (Mr., Mrs., etc)
    z = raw_X[i,1].split()[0] 
    if z == 'the':
        z = raw_X[i,1].split()[1]
    raw_X[i,1] = z
    # get the Ticket class
    z = raw_X[i,-3].split()[0]
    if z.isdigit(): #No Ticket class, just number
        raw_X[i,-3] = '0'
    elif z[0] == 'A': # A4 or A5
        if '4' in z:
            raw_X[i,-3] = 'A4'
        else:
            raw_X[i,-3] = 'A5'
    elif z[0] == 'C': # CA or C
        if 'A' in z:
            raw_X[i,-3] = 'CA'
        else:
            raw_X[i,-3] = 'C'
    elif z[0] == 'F' and 'C' in z: #FC or FCC
        raw_X[i,-3] = 'FC'
    elif z[0] == 'P': # PC or PP
        if z == 'PC':
            raw_X[i,-3] = 'PC'
        else:
            raw_X[i,-3] = 'PP'
    elif z[:3] == 'S.C.' or z[:1] == 'SC':
        raw_X[i,-3] = 'SC'
    elif z[:2] == 'S.O' or z[:2] == 'SO/':
        raw_X[i,-3] = 'SO'
    elif z[:4] == 'SOTON' or z[:3] == 'STON':
        raw_X[i,-3] = 'STON'
    elif z[0] == 'W': # WC or WEP
        if 'C' in z:
            raw_X[i,-3] = 'WC'
        else:
            raw_X[i,-3] = 'WEP'
    else:
        raw_X[i,-3] = 'other'
#to get the titles    
#unique, counts = np.unique(z, return_counts=True)
#zz = np.asarray((unique, counts)).T 
X = np.empty((raw_X.shape[0],raw_X.shape[1]+1))
for i in range(X.shape[0]):
    X[i,0] = int(raw_X[i,0]) #Pclass, or passenger class
    if raw_X[i,1] in ['Countess.','Lady.','Mlle.','Mme.']: #if title is female nobility
        X[i,1] = 0
    elif raw_X[i,1] in ['Don.','Jonkheer.','Sir.']: #if title is male nobility
        X[i,1] = 1
    elif raw_X[i,1] == 'Dr.': #if doctor
        X[i,1] = 2  
    elif raw_X[i,1] == 'Master.': #if child
        X[i,1] = 3        
    elif raw_X[i,1] in ['Mr.','Ms.','Miss.','Mrs.']: #if commoners
        X[i,1] = 4
    elif raw_X[i,1] in ['Capt.','Col.','Major.']: #if title is military
        X[i,1] = 5
    elif raw_X[i,1] == 'Rev.': #if clergy
        X[i,1] = 6      
    else:
        X[i,1] = -1
    X[i,2] = 1 if raw_X[i,2]=='male' else -1 #Sex, 1 if male, -1 if female
    try:
        X[i,3] = raw_X[i,3] #Age
    except ValueError:
        X[i,3] = np.nan #will impute later
    X[i,4] = int(raw_X[i,4]) #Sibsp, or number of siblings and spouses onboard
    X[i,5] = int(raw_X[i,5]) #Parch, or number of parents and children onbard
    if raw_X[i,6] == '0':
        X[i,6] = 0
    elif raw_X[i,6] == 'A4':
        X[i,6] = 1
    elif raw_X[i,6] == 'A5':
        X[i,6] = 2
    elif raw_X[i,6] == 'CA':
        X[i,6] = 3
    elif raw_X[i,6] == 'C':
        X[i,6] = 4
    elif raw_X[i,6] == 'FC':
        X[i,6] = 5   
    elif raw_X[i,6] == 'PC':
        X[i,6] = 6
    elif raw_X[i,6] == 'PP':
        X[i,6] = 7
    elif raw_X[i,6] == 'SC':
        X[i,6] = 8
    elif raw_X[i,6] == 'SO':
        X[i,6] = 9
    elif raw_X[i,6] == 'STON':
        X[i,6] = 10
    elif raw_X[i,6] == 'WC':
        X[i,6] = 11
    elif raw_X[i,6] == 'WEP':
        X[i,6] = 12
    else:
        X[i,6] = 13
    X[i,7] = float(raw_X[i,7]) #Fare paid
    if raw_X[i,8][0] == 'C':
        X[i,8] = 0 #0 means from Cherbourg
    elif raw_X[i,8][0] == 'Q':
        X[i,8] = 1 #1 means from Queenstown
    elif raw_X[i,8][0] == 'S':
        X[i,8] = 2 #2 means from Southampton
    else:
        X[i,8] = np.nan
    X[i,9] = X[i,4] + X[i,5] + 1# total family size
#fill in missing ages
Xchild = X[X[:,1] == 3]
median_child_age = np.nanmedian(Xchild[:,1])
Xadult = X[X[:,1] != 3]
median_adult_age = np.nanmedian(Xadult[:,1])
for i in range(X.shape[0]):
    if X[i,3] == np.nan:
        if X[i,1] == 3: #if child
            X[i,3] = median_child_age
        else:
            X[i,3] = median_adult_age
#fill in other missing values
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
feature_columns = {1: 'Pclass', 3: 'Title', 4: 'Sex', 5: 'Age', 6: 'SibSp', 7: 'Parch', 8: 'Ticket', 9: 'Fare', 11: 'Embarked'}
raw_X = raw_data[:,feature_columns.keys()]
for i in range(raw_X.shape[0]):
    # get the title (Mr., Mrs., etc)
    z = raw_X[i,1].split()[0] 
    if z == 'the':
        z = raw_X[i,1].split()[1]
    raw_X[i,1] = z
    # get the Ticket class
    z = raw_X[i,-3].split()[0]
    if z.isdigit(): #No Ticket class, just number
        raw_X[i,-3] = '0'
    elif z[0] == 'A': # A4 or A5
        if '4' in z:
            raw_X[i,-3] = 'A4'
        else:
            raw_X[i,-3] = 'A5'
    elif z[0] == 'C': # CA or C
        if 'A' in z:
            raw_X[i,-3] = 'CA'
        else:
            raw_X[i,-3] = 'C'
    elif z[0] == 'F' and 'C' in z: #FC or FCC
        raw_X[i,-3] = 'FC'
    elif z[0] == 'P': # PC or PP
        if z == 'PC':
            raw_X[i,-3] = 'PC'
        else:
            raw_X[i,-3] = 'PP'
    elif z[:3] == 'S.C.' or z[:1] == 'SC':
        raw_X[i,-3] = 'SC'
    elif z[:2] == 'S.O' or z[:2] == 'SO/':
        raw_X[i,-3] = 'SO'
    elif z[:4] == 'SOTON' or z[:3] == 'STON':
        raw_X[i,-3] = 'STON'
    elif z[0] == 'W': # WC or WEP
        if 'C' in z:
            raw_X[i,-3] = 'WC'
        else:
            raw_X[i,-3] = 'WEP'
    else:
        raw_X[i,-3] = 'other'
test_X = np.empty((raw_X.shape[0],raw_X.shape[1]+1))
for i in range(test_X.shape[0]):
    test_X[i,0] = int(raw_X[i,0]) #Pclass, or passenger class
    if raw_X[i,1] in ['Countess.','Lady.','Mlle.','Mme.']: #if title is female nobility
        test_X[i,1] = 0
    elif raw_X[i,1] in ['Don.','Jonkheer.','Sir.']: #if title is male nobility
        test_X[i,1] = 1
    elif raw_X[i,1] == 'Dr.': #if doctor
        test_X[i,1] = 2  
    elif raw_X[i,1] == 'Master.': #if child
        test_X[i,1] = 3        
    elif raw_X[i,1] in ['Mr.','Ms.','Miss.','Mrs.']: #if commoners
        test_X[i,1] = 4
    elif raw_X[i,1] in ['Capt.','Col.','Major.']: #if title is military
        test_X[i,1] = 5
    elif raw_X[i,1] == 'Rev.': #if clergy
        test_X[i,1] = 6      
    else:
        test_X[i,1] = -1
    test_X[i,2] = 1 if raw_X[i,2]=='male' else -1 #Sex, 1 if male, -1 if female
    try:
        test_X[i,3] = float(raw_X[i,3]) #Age
    except ValueError:
        test_X[i,3] = np.nan
    test_X[i,4] = int(raw_X[i,4]) #Sibsp, or number of siblings and spouses onboard
    test_X[i,5] = int(raw_X[i,5]) #Parch, or number of parents and children onbard
    if raw_X[i,6] == '0':
        test_X[i,6] = 0
    elif raw_X[i,6] == 'A4':
        test_X[i,6] = 1
    elif raw_X[i,6] == 'A5':
        test_X[i,6] = 2
    elif raw_X[i,6] == 'CA':
        test_X[i,6] = 3
    elif raw_X[i,6] == 'C':
        test_X[i,6] = 4
    elif raw_X[i,6] == 'FC':
        test_X[i,6] = 5   
    elif raw_X[i,6] == 'PC':
        test_X[i,6] = 6
    elif raw_X[i,6] == 'PP':
        test_X[i,6] = 7
    elif raw_X[i,6] == 'SC':
        test_X[i,6] = 8
    elif raw_X[i,6] == 'SO':
        test_X[i,6] = 9
    elif raw_X[i,6] == 'STON':
        test_X[i,6] = 10
    elif raw_X[i,6] == 'WC':
        test_X[i,6] = 11
    elif raw_X[i,6] == 'WEP':
        test_X[i,6] = 12
    else:
        test_X[i,6] = 13
    try:
        test_X[i,7] = float(raw_X[i,7]) #Fare paid
    except ValueError:
        test_X[i,7] = np.nan
    if raw_X[i,8][0] == 'C':
        test_X[i,8] = 0 #0 means from Cherbourg
    elif raw_X[i,8][0] == 'Q':
        test_X[i,8] = 1 #1 means from Queenstown
    elif raw_X[i,8][0] == 'S':
        test_X[i,8] = 2 #2 means from Southampton
    else:
        test_X[i,8] = np.nan
    test_X[i,9] = test_X[i,4] + test_X[i,5] + 1 # total family size
#fill in missing ages
Xchild = test_X[test_X[:,1] == 3]
median_child_age = np.nanmedian(Xchild[:,1])
Xadult = test_X[test_X[:,1] != 3]
median_adult_age = np.nanmedian(Xadult[:,1])
for i in range(test_X.shape[0]):
    if test_X[i,3] == np.nan:
        if test_X[i,1] == 3: #if child
            test_X[i,3] = median_child_age
        else:
            test_X[i,3] = median_adult_age
#fill in other missing values
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
    #features = ['Pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'boarding']
    features = list(feature_columns.values()) + ['familysize']
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