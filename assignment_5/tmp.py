from sklearn import svm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import precision_recall_fscore_support

import numpy as np

# Load datasets
from sklearn import datasets
cancer = datasets.load_breast_cancer()
print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)
print(cancer.data.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data,
    cancer.target,
    test_size=0.3,
    random_state=77
)

clf_poly = svm.SVC(
    kernel='poly',
    degree=2,
    gamma='auto'
)

clf_rbf = svm.SVC(kernel='rbf')

clf_poly.fit(X_train, y_train)
clf_rbf.fit(X_train, y_train)

def get_accuracy_precision_recall_f_score(clf_instance, X_train, y_train, X_test, y_test):
    print(X_train.shape)
    print(X_train)
    print(y_train.shape)
    print(y_train)
    print(clf_instance)
    clf_instance.fit(
        X_train,
        y_train
    )

    y_pred = clf_instance.predict(X_test)

    accuracy = clf_instance.score(X_test, y_test)

    precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred)

    return accuracy, precision, recall, f_score

def plot_svm(hyperparameters, X_train, y_train, X_test, y_test, polynomials=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    list_accuracy = []
    list_precision = []
    list_recall = []
    list_f_score = []

    if polynomials:
        for gamma in hyperparameters['gamma']:
            for degree in hyperparameters['degree']:

                clf_poly = svm.SVC(
                    kernel='poly',
                    degree=hyperparameters['degree'],
                    gamma=hyperparameters['gamma']
                )

                accuracy, precision, recall, f_score = get_accuracy_precision_recall_f_score(
                    clf_poly,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )

                list_accuracy.append(accuracy)
                list_precision.append(precision)
                list_recall.append(recall)
                list_f_score.append(f_score)

        Axes3D.plot_surface(hyperparameters['gamma'], hyperparameters['degree'], list_accuracy)
        Axes3D.plot_surface(hyperparameters['gamma'], hyperparameters['degree'], list_precision)
        Axes3D.plot_surface(hyperparameters['gamma'], hyperparameters['degree'], list_recall)
        Axes3D.plot_surface(hyperparameters['gamma'], hyperparameters['degree'], list_f_score)

    else:
        for gamma in hyperparameters['gamma']:
            for penalty_c in hyperparameters['penalty_c']:

                clf_rbf = svm.SVC(
                    kernel='rbf',
                    degree=hyperparameters['gamma'],
                    C=hyperparameters['penalty_c']
                )

                accuracy, precision, recall, f_score = get_accuracy_precision_recall_f_score(
                    clf_rbf,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )

                list_accuracy.append(accuracy)
                list_precision.append(precision)
                list_recall.append(recall)
                list_f_score.append(f_score)

        Axes3D.plot_surface(hyperparameters['gamma'], hyperparameters['penalty_c'], list_accuracy)
        Axes3D.plot_surface(hyperparameters['gamma'], hyperparameters['penalty_c'], list_precision)
        Axes3D.plot_surface(hyperparameters['gamma'], hyperparameters['penalty_c'], list_recall)
        Axes3D.plot_surface(hyperparameters['gamma'], hyperparameters['penalty_c'], list_f_score)

list_gamma = ['scale', 'auto', 'auto_deprecated']

hyperparameters = {
    'gamma': list_gamma,
    'degree': [0, 1, 2, 3, 4, 5, 6]
}
plot_svm(hyperparameters, X_train, y_train, X_test, y_test, polynomials=True)

hyperparameters = {
    'gamma': list_gamma,
    'penalty_c': [0.1, 1, 10, 100, 1000]
}
plot_svm(hyperparameters, X_train, y_train, X_test, y_test, polynomials=False)
