import numpy as np
import joblib
from xgboost import XGBClassifier, XGBRegressor

# TODO pohrat si s hyperparametry - i co nejsou zatim zmineny
# - GridSearchCV

# GPU
# gpu_params = {
#     "tree_method": "hist",
#     "device": "cuda",
# }

# ------XGBOOST REGG-------
def train_xgbr(X_train, X_test, y_train, obj='reg:squarederror', save=False, n_estimators=500, max_depth=5, l_rate=0.1):
    print("-----TRAIN XGBoost Regressor-----")

    xgbr = XGBRegressor(objective=obj, n_estimators=n_estimators, max_depth=max_depth, learning_rate=l_rate, random_state=42, n_jobs=-1)
    xgbr.fit(X_train, y_train)

    y_pred = xgbr.predict(X_test)
    y_pred = np.array(y_pred)

    if save:
        joblib.dump(xgbr, "../trained_models/xgbr_last.pkl")

    print("-----FINISHED XGBoost Regressor-----")

    return y_pred



# ------XGBOOST CLASS-------
def train_xgbc(X_train, X_test, y_train, obj='binary:logistic', l_rate=0.1, save=False, n_estimators=500, max_depth=5):
    print("-----TRAIN XGBoost Classifier-----")

    xgbc = XGBClassifier(objective=obj, n_estimators=n_estimators, max_depth=max_depth, learning_rate=l_rate, random_state=42, n_jobs=-1)
    xgbc.fit(X_train, y_train)

    y_pred_proba = xgbc.predict_proba(X_test)[:, 1]

    if save:
        joblib.dump(xgbc, "../trained_models/xgbc_last.pkl")

    print("-----FINISHED XGBoost Classifier-----")

    return y_pred_proba

