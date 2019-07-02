import matplotlib.pyplot as plt
from numpy import shape, trapz, array
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':

    '''
    - load the breast cancer dataset using the sklearn.datasets module;
    - import the learning algorithm of your choice (e.g., `linear_model.LogisticRegression`);
    - use the fit method to learn a new classifier specifying that you want the classifier to be setup to output probability estimates (use parameter 'probability=True' when instantiating the class);
    - use the learnt classifier to obtain the scores for the objects in the training set (use the `predict_proba` method);
    '''

    cancer_dataset = datasets.load_breast_cancer(True)
    print(f"Shape of cancer dataset data: {shape(cancer_dataset[0])}")
    print(f"Shape of cancer dataset targets: {shape(cancer_dataset[1])}")

    clf = LogisticRegression(solver="liblinear")

    folds = 5

    scores = cross_val_score(clf, X=cancer_dataset[0], y=cancer_dataset[1], cv=folds)
    print(f"\nCross validation accuracy scores with {folds} folds: {scores}")

    clf.fit(cancer_dataset[0], cancer_dataset[1])
    probability_estimates = clf.predict_proba(cancer_dataset[0])
    print(f"\nScores on on the test set {probability_estimates}")

    '''
    - Use the scores you just obtained to rank the examples (from higher to lower probability of being in class 1);
    - consider all possible classifiers you can obtain from such order by splitting the sequence in two and then 
        deciding to label everything on the left as positive and everything on the right as negative);
    - evaluate the number of false positive examples (FP) and the number of true positive examples (TP) for each split;
    - plot those values on a scatter plot (hint: use the `matplotlib.pyplot.plot` function);
    '''

    n_pos_examples = cancer_dataset[1].tolist().count(0)
    n_neg_examples = cancer_dataset[1].tolist().count(1)

    print(f"\nPositive examples into the dataset: {n_pos_examples}")
    print(f"Negative examples into the dataset: {n_neg_examples}")

    dataset_information = []

    for i in range(0, len(cancer_dataset[0])):
        dataset_information.append((probability_estimates[i], cancer_dataset[0][i], cancer_dataset[1][i]))

    dataset_information.sort(key=lambda tuple: tuple[0][0], reverse=True)

    classifiers_by_threshold = []

    dataset_length = len(dataset_information)

    for i in range(0, dataset_length):
        result = dict()
        result["threshold"] = dataset_information[i][0][0]
        result["tn"] = 0
        result["fn"] = 0
        result["tp"] = 0
        result["fp"] = 0
        result["pos"] = 0
        result["neg"] = 0

        for j in range(0, i):
            result["pos"] += 1
            if dataset_information[j][2] == 0:
                result["tp"] += 1
            else:
                result["fp"] += 1

        for j in range(i, dataset_length):
            result["neg"] += 1
            if dataset_information[j][2] != 0:
                result["tn"] += 1
            else:
                result["fn"] += 1

        classifiers_by_threshold.append(result)

    plot_coords = ([], [])

    for element in classifiers_by_threshold:
        plot_coords[0].append(element["fp"])
        plot_coords[1].append(element["tp"])

    plt.plot(plot_coords[0], plot_coords[1])
    plt.xlabel("False Positive (FP)")
    plt.ylabel("True Positive (TP)")
    plt.tight_layout()
    plt.show()

    print(f"ROC AUC = {trapz(array(plot_coords[1]), array(plot_coords[0]))}")

    positive_scores = []

    for i in range(0, len(probability_estimates)):
        positive_scores.insert(0, probability_estimates[i][0])

    fpr, tpr, thresholds = roc_curve(cancer_dataset[1], positive_scores, pos_label=2)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive (FP)")
    plt.ylabel("True Positive (TP)")
    plt.tight_layout()
    plt.show()
