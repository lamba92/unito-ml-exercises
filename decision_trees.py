from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
import graphviz

if __name__ == '__main__':
    iris = load_iris()
    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5)
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.8, random_state=300)
    clf.fit(x_train, y_train)

    for i in range(0, len(y_test)):
        prediction = clf.predict([x_test[i]])[0]
        prediction_test = "CORRECT"
        if prediction != y_test[i]:
            prediction_test = "ERROR"
        print(f"Instance #{i} | {prediction_test}\n - predicted: {iris.target_names[prediction]}\n"
              f" - actual:    {iris.target_names[y_test[i]]}\n")

    predicted_y_test = clf.predict(x_test)

    accuracy = accuracy_score(y_test, predicted_y_test)
    f1_score = f1_score(y_test, predicted_y_test, average="macro")

    print(f"Overall accuracy: {accuracy} | Overall f1 score: {f1_score}")

    n_folds = 5

    accuracy_scores_k_fold = cross_val_score(clf, iris.data, iris.target, cv=n_folds)

    print(f"{n_folds} folds cross validation accuracies: {accuracy_scores_k_fold}\n")

    # Only in Jupyter notebooks!
    # dot_data = export_graphviz(
    #     decision_tree=clf,
    #     out_file=None,
    #     feature_names=iris.feature_names,
    #     class_names=iris.target_names,
    #     filled=True,
    #     rounded=True,
    #     special_characters=True
    # )
    # graph = graphviz.Source(dot_data)
    # graph

    text_graph = export_text(
        decision_tree=clf,
        feature_names=iris.feature_names
    )
    print("Graph visualization:")
    print(text_graph)
