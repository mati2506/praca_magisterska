import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
import myMLP
import myNetworkPruning as prune




   

#test działania
data = pd.read_csv("raw_data/iris.data")
X = data.iloc[:,0:4].values
Y = data.iloc[:,4].values
X_train, X_test, y_train, y_test = train_test_split(X, Y)
clf = myMLP.Classifier(epochs=100, activation="sigmoid")
clf.fit(X_train, y_train)
#print(clf.coefs_)
print(accuracy_score(y_train, clf.predict(X_train)))

ll = 0.05

#clf1 = copy.deepcopy(clf)
#a, d1 = prune.simple_pruning(clf1, ll, X_train, y_train)
#print(a)
#print(d1)
##print(clf1.coefs_)
#print(clf1.hidden)

#print(accuracy_score(y_train, clf1.predict(X_train)))
#print(accuracy_score(y_test, clf1.predict(X_test)))
#print()


#clf2 = copy.deepcopy(clf)
#b, d2 = prune.simple_pruning_amendment(clf2, ll, X_train, y_train)
#print(b)
#print(d2)
##print(clf2.coefs_)
#print(clf2.hidden)

#print(accuracy_score(y_train, clf2.predict(X_train)))
#print(accuracy_score(y_test, clf2.predict(X_test)))
#print()


#clf3 = copy.deepcopy(clf)
#c, d3 = prune.karnin_pruning(clf3, ll, X_train, y_train)
#print(c)
#print(d3)
##print(clf3.coefs_)
#print(clf3.hidden)

#print(accuracy_score(y_train, clf3.predict(X_train)))
#print(accuracy_score(y_test, clf3.predict(X_test)))
#print()


#clf4 = copy.deepcopy(clf)
#d, d4 = prune.pruning_by_variance(clf4, ll, X_train, y_train)
#print(d)
#print(d4)
##print(clf4.coefs_)
#print(clf4.hidden)

#print(accuracy_score(y_train, clf4.predict(X_train)))
#print(accuracy_score(y_test, clf4.predict(X_test)))
#print()


clf5 = copy.deepcopy(clf)
e, d5 = prune.FBI_pruning(clf5, ll, X_train, y_train)
print(e)
print(d5)
#print(clf5.coefs_)
print(clf5.hidden)

print(accuracy_score(y_train, clf5.predict(X_train)))
print(accuracy_score(y_test, clf5.predict(X_test)))
print()


clf6 = copy.deepcopy(clf)
f, d6 = prune.APERT_pruning(clf6, ll, X_train, y_train)
print(f)
print(d6)
#print(clf6.coefs_)
print(clf6.hidden)

print(accuracy_score(y_train, clf6.predict(X_train)))
print(accuracy_score(y_test, clf6.predict(X_test)))
print()


clf7 = copy.deepcopy(clf)
g, d7 = prune.APERTP_pruning(clf7, ll, X_train, y_train)
print(g)
print(d7)
#print(clf7.coefs_)
print(clf7.hidden)

print(accuracy_score(y_train, clf7.predict(X_train)))
print(accuracy_score(y_test, clf7.predict(X_test)))
print()


clf8 = copy.deepcopy(clf)
h, d8 = prune.PD_pruning(clf8, ll, X_train, y_train)
print(h)
print(d8)
#print(clf8.coefs_)
print(clf8.hidden)

print(accuracy_score(y_train, clf8.predict(X_train)))
print(accuracy_score(y_test, clf8.predict(X_test)))
print()


clf9 = copy.deepcopy(clf)
i, d9 = prune.PD_pruning(clf9, ll, X_train, y_train)
print(i)
print(d9)
#print(clf9.coefs_)
print(clf9.hidden)

print(accuracy_score(y_train, clf9.predict(X_train)))
print(accuracy_score(y_test, clf9.predict(X_test)))
print()



x = np.sort(np.random.uniform(-2,2,20)).reshape(-1,1)
y = 2*x + 1

reg = myMLP.Regressor(activation="sigmoid")
reg.fit(x,y)
#print(reg.coefs_)
print(mean_squared_error(y, reg.predict(x)))

reg1 = copy.deepcopy(reg)
aa, dd1 = prune.PEB_pruning(reg1, 0.15, x, y)
print(aa)
print(dd1)
print(reg1.hidden)

print(mean_squared_error(y, reg1.predict(x)))