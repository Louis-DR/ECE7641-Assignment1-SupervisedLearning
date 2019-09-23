#%%
print("----------Imports----------")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import importlib
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-ticks')
import io
import pydotplus
import graphviz

from scraper import import_wine_review, import_wine_quality
from extimecalc import chrono
from analysis import plot_confusion_matrix, plot_param_crossval_acc, analyse

verbose = True
print("----------Done importing----------")



#region[white]
#%%
print("----------Loading dataset----------")
XXX, yyy = import_wine_quality(subset_size=0)
X_train, X_test, y_train, y_test = train_test_split(XXX, yyy, test_size=0.15, random_state=0)
fold = 4
print("----------Done loading dataset----------")
#endregion



#region[purple] DECISION TREE
#%%
if (verbose) : print("\n\n----------Best parameters----------")

clf = tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=10, max_depth=4)
# clf = tree.DecisionTreeClassifier(criterion='entropy')

clf = analyse(clf, fold, XXX, yyy, X_train, y_train, X_test, y_test)

clf.fit(XXX,yyy)
print("The depth is : {}".format(clf.get_depth()))
print("The number of leaves is : {}".format(clf.get_n_leaves()))
dot_data = io.StringIO()
tree.export_graphviz(clf, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("DecisionTree_quality.pdf")



#%%
if (verbose) : print("\n\n----------Maximum depth----------")
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 30]

acc_min = []
acc_avg = []
acc_max = []
acc_train = []
acc_test = []

chrono(prt=False)

for md in max_depths :
    if (verbose) : print("For a maximum depth of {}".format(md))

    # clf = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=80, max_depth=md)
    clf = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=md)
    scores = cross_val_score(clf, XXX, yyy, cv=fold)
    clf.fit(XXX,yyy)
    accuracy = clf.score(XXX,yyy)
    clf.fit(X_train,y_train)
    accuracy_test = clf.score(X_test,y_test)

    acc_min.append(min(scores))
    acc_avg.append(sum(scores)/len(scores))
    acc_max.append(max(scores))
    acc_train.append(accuracy)
    acc_test.append(accuracy_test)

    if (verbose) : print("     the depth is : {}".format(clf.get_depth()))
    if (verbose) : print("     the number of leaves is : {}".format(clf.get_n_leaves()))
    if (verbose) : print("     the training accuracy is : {0:.3f}".format(accuracy))
    if (verbose) : print("     the testing accuracy is : {0:.3f}".format(accuracy_test))
    if (verbose) : print("     the cross validation accuracy is : {0:.3f}".format(sum(scores)/len(scores)))
    chrono(5, prt=verbose)

plot_param_crossval_acc(max_depths, acc_train, acc_test, acc_min, acc_avg, acc_max, xlabel="Maximum depth")
plt.show()



#%%
if (verbose) : print("\n\n----------Maximum leaves----------")
max_leaf_nodess = [2, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 5000, 10000]

acc_min = []
acc_avg = []
acc_max = []
acc_train = []
acc_test = []

chrono(prt=False)

for ml in max_leaf_nodess :
    if (verbose) : print("For a maximum of {} leaves".format(ml))

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=ml)
    scores = cross_val_score(clf, XXX, yyy, cv=fold)
    clf.fit(XXX,yyy)
    accuracy = clf.score(XXX,yyy)
    clf.fit(X_train,y_train)
    accuracy_test = clf.score(X_test,y_test)

    acc_min.append(min(scores))
    acc_avg.append(sum(scores)/len(scores))
    acc_max.append(max(scores))
    acc_train.append(accuracy)
    acc_test.append(accuracy_test)

    if (verbose) : print("     the depth is : {}".format(clf.get_depth()))
    if (verbose) : print("     the number of leaves is : {}".format(clf.get_n_leaves()))
    if (verbose) : print("     the training accuracy is : {0:.3f}".format(accuracy))
    if (verbose) : print("     the testing accuracy is : {0:.3f}".format(accuracy_test))
    if (verbose) : print("     the cross validation accuracy is : {0:.3f}".format(sum(scores)/len(scores)))
    chrono(5, prt=verbose)

plot_param_crossval_acc(max_leaf_nodess, acc_train, acc_test, acc_min, acc_avg, acc_max, xlabel="Maximum leaves", log=True)
plt.show()


#endregion



#region[red] BOOSTING
#%%
if (verbose) : print("\n\n----------Best parameters----------")

clf = tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=10, max_depth=4)
# clf = tree.DecisionTreeClassifier(criterion='entropy')

ada = AdaBoostClassifier(clf, learning_rate=0.01, n_estimators=40)

ada = analyse(ada, fold, XXX, yyy, X_train, y_train, X_test, y_test)



#%%
if (verbose) : print("\n\n----------Learning rates----------")
# learning_rates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
learning_rates = learning_rates[::-1]

acc_min = []
acc_avg = []
acc_max = []
acc_train = []
acc_test = []

chrono(prt=False)

clf = tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=10, max_depth=4)
scores = cross_val_score(clf, XXX, yyy, cv=fold)
clf_acc = sum(scores)/len(scores)
if (verbose) : print("The tree accuracy is : {0:.3f}".format(clf_acc))

for lr in learning_rates :
    if (verbose) : print("For a learning rate of {}".format(lr))

    ada = AdaBoostClassifier(clf, learning_rate=lr, n_estimators=40)
    scores = cross_val_score(ada, XXX, yyy, cv=fold)
    ada.fit(XXX,yyy)
    accuracy = ada.score(XXX,yyy)
    ada.fit(X_train,y_train)
    accuracy_test = ada.score(X_test,y_test)

    acc_min.append(min(scores))
    acc_avg.append(sum(scores)/len(scores))
    acc_max.append(max(scores))
    acc_train.append(accuracy)
    acc_test.append(accuracy_test)

    if (verbose) : print("     the training accuracy is : {0:.3f}".format(accuracy))
    if (verbose) : print("     the testing accuracy is : {0:.3f}".format(accuracy_test))
    if (verbose) : print("     the cross validation accuracy is : {0:.3f}".format(sum(scores)/len(scores)))
    chrono(5, prt=verbose)

plot_param_crossval_acc(learning_rates, acc_train, acc_test, acc_min, acc_avg, acc_max, xlabel="Learning rate", log=True)
plt.show()



#%%
if (verbose) : print("\n\n----------Number of estimators----------")
# n_estimatorss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
n_estimatorss = [10,20,30,40,50,60,70, 80, 90,100]
# n_estimatorss = [1, 3, 10, 35, 100, 350, 1000, 3500]

acc_min = []
acc_avg = []
acc_max = []
acc_train = []
acc_test = []

chrono(prt=False)

clf = tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=10, max_depth=4)
scores = cross_val_score(clf, XXX, yyy, cv=fold)
clf_acc = sum(scores)/len(scores)
if (verbose) : print("The tree accuracy is : {0:.3f}".format(clf_acc))

for ne in n_estimatorss :
    if (verbose) : print("For {} estimators".format(ne))

    ada = AdaBoostClassifier(clf,learning_rate=0.01, n_estimators=ne)
    scores = cross_val_score(ada, XXX, yyy, cv=fold)
    ada.fit(XXX,yyy)
    accuracy = ada.score(XXX,yyy)
    ada.fit(X_train,y_train)
    accuracy_test = ada.score(X_test,y_test)

    acc_min.append(min(scores))
    acc_avg.append(sum(scores)/len(scores))
    acc_max.append(max(scores))
    acc_train.append(accuracy)
    acc_test.append(accuracy_test)

    if (verbose) : print("     the training accuracy is : {0:.3f}".format(accuracy))
    if (verbose) : print("     the testing accuracy is : {0:.3f}".format(accuracy_test))
    if (verbose) : print("     the cross validation accuracy is : {0:.3f}".format(sum(scores)/len(scores)))
    chrono(5, prt=verbose)

plot_param_crossval_acc(n_estimatorss, acc_train, acc_test, acc_min, acc_avg, acc_max, xlabel="Number of estimators", log=False)
plt.show()

#endregion



#region[yellow]
#%%
if (verbose) : print("\n\n----------Best parameters----------")

neigh = KNeighborsClassifier(n_neighbors=200, weights='distance', metric='euclidean')
neigh = analyse(neigh, fold, XXX, yyy, X_train, y_train, X_test, y_test)



#%%
if (verbose) : print("\n\n----------Number of neighbours----------")
# n_neighborss = [1, 2, 3, 4, 5, 7, 10]
n_neighborss = [1, 3, 10, 35, 100, 180, 300, 400, 600, 1000, 3500]

acc_min = []
acc_avg = []
acc_max = []
acc_train = []
acc_test = []

chrono(prt=False)

for n_neighbors in n_neighborss :
    if (verbose) : print("For {} neighbours".format(n_neighbors))

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='chebyshev')
    scores = cross_val_score(neigh, XXX, yyy, cv=fold)
    neigh.fit(XXX,yyy)
    accuracy = neigh.score(XXX,yyy)
    neigh.fit(X_train,y_train)
    accuracy_test = neigh.score(X_test,y_test)

    acc_min.append(min(scores))
    acc_avg.append(sum(scores)/len(scores))
    acc_max.append(max(scores))
    acc_train.append(accuracy)
    acc_test.append(accuracy_test)

    if (verbose) : print("     the training accuracy is : {0:.3f}".format(accuracy))
    if (verbose) : print("     the testing accuracy is : {0:.3f}".format(accuracy_test))
    if (verbose) : print("     the cross validation accuracy is : {0:.3f}".format(sum(scores)/len(scores)))
    chrono(5, prt=verbose)

plot_param_crossval_acc(n_neighborss, acc_train, acc_test, acc_min, acc_avg, acc_max, xlabel="Number of neighbours", log=True)
plt.show()

#endregion



#region[green] SVM
#%%
if (verbose) : print("\n\n----------Best parameters----------")

svm = SVC(kernel='linear', gamma='auto', C=3.5)
svm = analyse(svm, fold, XXX, yyy, X_train, y_train, X_test, y_test)


#%%
if (verbose) : print("\n\n----------Kernel----------")

chrono(prt=False)

kernels = ['rbf', 'sigmoid', 'linear']
for ker in kernels:
    print("For the kernel {} :".format(ker))

    svm = SVC(kernel=ker, gamma='auto')
    svm = analyse(svm, fold, XXX, yyy, X_train, y_train, X_test, y_test)

    chrono(5, prt=verbose)

#%%
if (verbose) : print("\n\n----------Kernel poly----------")

chrono(prt=False)

for deg in range(5):
    print("For the kernel poly of order {} :".format(deg))

    svm = SVC(kernel='poly', degree=deg, gamma='auto')
    svm = analyse(svm, fold, XXX, yyy, X_train, y_train, X_test, y_test)

    chrono(5, prt=verbose)



#%%
if (verbose) : print("\n\n----------Penalty----------")
cccs = [0.035, 0.1, 0.35, 1.0, 3.5, 10, 35, 100, 350]

acc_min = []
acc_avg = []
acc_max = []
acc_train = []
acc_test = []

chrono(prt=False)

for ccc in cccs :
    if (verbose) : print("For a penalty factor of {}".format(ccc))

    svm = SVC(kernel='linear', gamma='auto', C=ccc)
    scores = cross_val_score(svm, XXX, yyy, cv=fold)
    svm.fit(XXX,yyy)
    accuracy = svm.score(XXX,yyy)
    svm.fit(X_train,y_train)
    accuracy_test = svm.score(X_test,y_test)

    acc_min.append(min(scores))
    acc_avg.append(sum(scores)/len(scores))
    acc_max.append(max(scores))
    acc_train.append(accuracy)
    acc_test.append(accuracy_test)

    if (verbose) : print("     the training accuracy is : {0:.3f}".format(accuracy))
    if (verbose) : print("     the testing accuracy is : {0:.3f}".format(accuracy_test))
    if (verbose) : print("     the cross validation accuracy is : {0:.3f}".format(sum(scores)/len(scores)))
    chrono(5, prt=verbose)

plot_param_crossval_acc(cccs, acc_train, acc_test, acc_min, acc_avg, acc_max, xlabel="C factor", log=True)
plt.show()



#endregion



#region[blue] NEURAL NETWORK
#%%
if (verbose) : print("\n\n----------Best parameters----------")

chrono(prt=False)
network = (12,12,12)
print("For the network {}".format(network))
nn = MLPClassifier(hidden_layer_sizes=network, activation='relu', verbose=False, max_iter=4000)
nn = analyse(nn, fold, XXX, yyy, X_train, y_train, X_test, y_test)
chrono(5, prt=verbose)



#%%

networks = [(3,3), (3,3,3), (3,3,3,3), (3,3,3,3,3), (3,3,3,3,3,3), (3,3,3,3,3,3,3)]
# networks = [(4,4), (4,4,4), (4,4,4,4)]
# networks = [(5,5), (5,5,5), (5,5,5,5)]
# networks = [(6,6), (6,6,6), (6,6,6,6)]

chrono(prt=False)

for network in networks:
    if (verbose) : print("\n\nFor the network {}".format(network))

    nn = MLPClassifier(hidden_layer_sizes=network, verbose=False, max_iter=4000)
    nn = analyse(nn, fold, XXX, yyy, X_train, y_train, X_test, y_test)

    chrono(5, prt=verbose)

#%%
networks = [(3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (12,12), (13,13), (14,14), (15,15), (16,16), (17,17), (18,18), (19,19), (20,20)]

acc_min = []
acc_avg = []
acc_max = []
acc_train = []
acc_test = []

chrono(prt=False)

for network in networks :
    if (verbose) : print("\n\nFor the network {}".format(network))

    nn = MLPClassifier(hidden_layer_sizes=network, verbose=False, max_iter=4000)
    scores = cross_val_score(nn, XXX, yyy, cv=fold)
    nn.fit(XXX,yyy)
    accuracy = nn.score(XXX,yyy)
    nn.fit(X_train,y_train)
    accuracy_test = nn.score(X_test,y_test)

    acc_min.append(min(scores))
    acc_avg.append(sum(scores)/len(scores))
    acc_max.append(max(scores))
    acc_train.append(accuracy)
    acc_test.append(accuracy_test)

    if (verbose) : print("     the training accuracy is : {0:.3f}".format(accuracy))
    if (verbose) : print("     the testing accuracy is : {0:.3f}".format(accuracy_test))
    if (verbose) : print("     the cross validation accuracy is : {0:.3f}".format(sum(scores)/len(scores)))
    chrono(5, prt=verbose)

network_height = [3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20]
plot_param_crossval_acc(network_height, acc_train, acc_test, acc_min, acc_avg, acc_max, xlabel="Number of layers")
plt.show()



#endregion


#%%
