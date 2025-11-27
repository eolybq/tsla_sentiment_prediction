import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# TODO pohrat si s hyperparametry - i co nejsou zatim zmineny
# - GridSearchCV


# -----DECISION TREE REGG------
def train_dtr(X_train, X_test, y_train, crit='squared_error', save=False, max_depth=5):
    print("-----TRAIN Decision Tree Regressor-----")

    dtr = DecisionTreeRegressor(criterion=crit, max_depth=max_depth, random_state=42)
    dtr.fit(X_train, y_train)

    y_pred = dtr.predict(X_test)
    y_pred = np.array(y_pred)

    if save:
        joblib.dump(dtr, "../trained_models/dtr_last.pkl")

    print("-----FINISHED Decision Tree Regressor-----")

    return y_pred


# -----DECISION TREE CLASS------
def train_dtc(X_train, X_test, y_train, pred_threshold, crit='gini', save=False, max_depth=5):
    print("-----TRAIN Decision Tree Classifier-----")

    dtc = DecisionTreeClassifier(criterion=crit, max_depth=max_depth, random_state=42)
    dtc.fit(X_train, y_train)

    y_pred_proba = dtc.predict_proba(X_test)[:, 1]

    y_pred_class = (y_pred_proba > pred_threshold).astype(int)

    if save:
        joblib.dump(dtc, "../trained_models/dtc_last.pkl")

    print("-----FINISHED Decision Tree Classifier-----")

    return y_pred_class, y_pred_proba



