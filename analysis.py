from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap='Blues'):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Ground truth',
           xlabel='Prediction')
    
    # Fix crop issue
    plt.ylim(-0.5, len(classes)-0.5)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_param_crossval_acc(params, acc_train, acc_test, acc_min, acc_avg, acc_max, xlabel="Parameter", log=False):
    fig, ax = plt.subplots()
    ax.plot(params, acc_train, color='indianred')
    ax.plot(params, acc_test, color='mediumseagreen')
    ax.plot(params, acc_avg, color='steelblue')
    ax.fill_between(params, acc_min, acc_max, color='steelblue', alpha='0.3')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    if log : plt.xscale("log")
    fig.tight_layout()
    return ax

def analyse(model, fold, XXX, yyy, X_train, y_train, X_test, y_test):
    scores = cross_val_score(model, XXX, yyy, cv=fold)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    accuracy_bal = balanced_accuracy_score(y_test, y_pred)
    train_sizes, train_scoress, test_scoress = learning_curve(model, XXX, yyy, cv=fold, n_jobs=-1, train_sizes=np.linspace(.05, 1.0, 12))
    print("The cross validation accuracy is : {0:.3f}".format(sum(scores)/len(scores)))
    print("The testing accuracy is : {0:.3f}".format(accuracy))
    print("The testing balanced accuracy is : {0:.3f}".format(accuracy_bal))

    train_mins = []
    train_avgs = []
    train_maxs = []
    for train_scores in train_scoress:
        train_mins.append(min(train_scores))
        train_avgs.append(sum(train_scores)/len(train_scores))
        train_maxs.append(max(train_scores))
    test_mins = []
    test_avgs = []
    test_maxs = []
    for test_scores in test_scoress:
        test_mins.append(min(test_scores))
        test_avgs.append(sum(test_scores)/len(test_scores))
        test_maxs.append(max(test_scores))
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_avgs, color='indianred')
    ax.fill_between(train_sizes, train_mins, train_maxs, color='indianred', alpha='0.3')
    ax.plot(train_sizes, test_avgs, color='mediumseagreen')
    ax.fill_between(train_sizes, test_mins, test_maxs, color='mediumseagreen', alpha='0.3')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning curves')
    plt.show()

    class_names = ['0', '1', '2', '3', '4']
    plot_confusion_matrix(y_test, y_pred, classes=class_names)
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True)
    plt.show()
    return model