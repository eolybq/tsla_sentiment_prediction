import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# TODO pohrat si s hyperparametry - i co nejsou zatim zmineny
# - GridSearchCV


# ------RANDOM FOREST REGG------
def train_rfr(X_train, X_test, y_train, crit='squared_error', save=False, n_estimators=500, max_depth=5):
    print("-----TRAIN Random Forest Regressor-----")

    rfr = RandomForestRegressor(n_estimators=n_estimators, criterion=crit, max_depth=max_depth, random_state=42, n_jobs=-1)
    rfr.fit(X_train, y_train)

    y_pred = rfr.predict(X_test)
    y_pred = np.array(y_pred)

    if save:
        joblib.dump(rfr, "../trained_models/rfr_last.pkl")

    print("-----FINISHED Random Forest Regressor-----")

    return y_pred



# ------RANDOM FOREST CLASS------
def train_rfc(X_train, X_test, y_train, crit='gini', save=False, n_estimators=500, max_depth=5):
    print("-----TRAIN Random Forest Classifier-----")

    rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=crit, max_depth=max_depth, random_state=42, n_jobs=-1)
    rfc.fit(X_train, y_train)

    y_pred_proba = rfc.predict_proba(X_test)[:, 1]

    if save:
        joblib.dump(rfc, "../trained_models/rfc_last.pkl")

    print("-----FINISHED Random Forest Classifier-----")

    return y_pred_proba

