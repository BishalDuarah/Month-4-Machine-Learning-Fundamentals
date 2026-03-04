import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    return metrics

def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return scores.mean()


def train_models(X_train, X_test, y_train, y_test):

    results = {}

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_model.fit(X_train, y_train)
    results["Logistic Regression"] = evaluate_model(log_model, X_test, y_test)

    # Decision Tree
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    results["Decision Tree"] = evaluate_model(tree_model, X_test, y_test)

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    results["Random Forest"] = evaluate_model(rf_model, X_test, y_test)

    return results, log_model, tree_model, rf_model