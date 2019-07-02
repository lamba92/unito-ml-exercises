from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":
    iris = load_iris()
    clf_tree = DecisionTreeClassifier(
        criterion="entropy",
        random_state=300,
        min_samples_leaf=5
    )

    n_neighbours = 11
    clf_knn = KNeighborsClassifier(
        n_neighbors=n_neighbours,
        weights="uniform"
    )

    n_folds = 5
    accuracy_scores_k_fold_tree = cross_val_score(
        clf_tree,
        iris.data,
        iris.target,
        cv=n_folds
    )
    accuracy_scores_k_fold_knn = cross_val_score(
        clf_knn,
        iris.data,
        iris.target,
        cv=n_folds
    )

    print(f"{n_folds} folds cross validation tree accuracies: {accuracy_scores_k_fold_tree}")
    print(f"{n_folds} folds cross validation knn accuracies:  {accuracy_scores_k_fold_knn}\n")

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

    plot_coord = ([], [], [])
    classifiers = []

    for k in range(1, len(y_train)):
        result = dict()
        result["k"] = k
        result["weights_distance"] = dict()
        result["weights_uniform"] = dict()
        
        result["weights_distance"]["clf"] = KNeighborsClassifier(k, weights="distance")
        result["weights_distance"]["clf"].fit(X_train, y_train)
        result["weights_distance"]["accuracy"] = accuracy_score(y_test, result["weights_distance"]["clf"].predict(X_test))
        
        result["weights_uniform"]["clf"] = KNeighborsClassifier(k, weights="uniform")
        result["weights_uniform"]["clf"].fit(X_train, y_train)
        result["weights_uniform"]["accuracy"] = accuracy_score(y_test, result["weights_uniform"]["clf"].predict(X_test))

        plot_coord[0].append(k)
        plot_coord[1].append(result["weights_distance"]["accuracy"])
        plot_coord[2].append(result["weights_uniform"]["accuracy"])
        classifiers.append(result)

    best_u_classifier = max(classifiers, key=lambda result: result["weights_uniform"]["accuracy"])
    best_d_classifier = max(classifiers, key=lambda result: result["weights_distance"]["accuracy"])

    print(f"\nBest number of neighbours by accuracy using uniform weights is:\n"
          f" - K = {best_u_classifier['k']}\n"
          f" - A: {best_u_classifier['weights_distance']['accuracy']}")
    print(f"Best number of neighbours by accuracy using distance based weights is:\n"
          f" - K = {best_d_classifier['k']}\n"
          f" - A: {best_d_classifier['weights_uniform']['accuracy']}")
    
    plt.plot(plot_coord[0], plot_coord[1], label="using distance as weights")
    plt.plot(plot_coord[0], plot_coord[2], label="using uniform as weights")
    plt.legend(loc='lower left')
    plt.xlabel("Number of neighbours")
    plt.ylabel("Accuracy")
    plt.show()
