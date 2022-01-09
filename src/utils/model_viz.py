import matplotlib.pyplot as plt
import seaborn as sns


def summary_of_model(clf, X_train, X_test, y_train, y_test, threshold):
    # from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score, recall_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve

    # This provides a summary of the model given a certain decision threshold of the predicted probability.
    # It includes a summary on recall/accuracy on the training and test sets, a visual display of the confusion matrix
    # and a plot of the precision-recall curve for a given classifier.
    pred_proba_test = clf.predict_proba(X_test)
    pred_test = (pred_proba_test[:, 1] >= threshold).astype("int")
    pred_proba_train = clf.predict_proba(X_train)
    pred_train = (pred_proba_train[:, 1] >= threshold).astype("int")
    print(classification_report(y_test, pred_test))
    
    print("Accuracy on the test set: {:.2f}".format(accuracy_score(y_test, pred_test)))
    print(confusion_matrix(y_test, pred_test))
    _, ax = plt.subplots(figsize=(9, 9))
    ax = sns.heatmap(
        confusion_matrix(y_test, pred_test),
        annot=True,
        fmt="d",
        cmap="vlag",
        annot_kws={"size": 40, "weight": "bold"},
    )
    labels = ["False", "True"]
    ax.set_xticklabels(labels, fontsize=25)
    ax.set_yticklabels(labels, fontsize=25)
    ax.set_ylabel("Actual", fontsize=30)
    ax.set_xlabel("Prediction", fontsize=30)
    lr_probs = clf.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    plt.figure()
    plt.plot(lr_recall, lr_precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
