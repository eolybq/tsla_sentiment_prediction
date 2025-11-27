import numpy as np
import joblib
from lightgbm import LGBMRegressor, LGBMClassifier

# TODO pohrat si s hyperparametry - i co nejsou zatim zmineny
# - GridSearchCV

# GPU
# gpu_params = {
#     'device': 'gpu',
#     'gpu_platform_id': 0,
#     'gpu_device_id': 0
# }


# -----LIGHTGBM REGG-------
def train_lgbr(X_train, X_test, y_train, save=False, n_estimators=500, max_depth=5, l_rate=0.1):
    print("-----TRAIN LightGBM Regressor-----")

    lgbr = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=l_rate, random_state=42, n_jobs=-1)
    lgbr.fit(X_train, y_train)

    y_pred = lgbr.predict(X_test)
    y_pred = np.array(y_pred)

    if save:
        joblib.dump(lgbr, "../trained_models/lgbr_last.pkl")

    print("-----FINISHED LightGBM Regressor-----")

    return y_pred



# -----LIGHTGBM CLASS-------
def train_lgbc(X_train, X_test, y_train, save=False, n_estimators=500, max_depth=5, l_rate=0.1):
    print("-----TRAIN LightGBM Classifier-----")

    lgbc = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=l_rate, random_state=42, n_jobs=-1)
    lgbc.fit(X_train, y_train)

    y_pred_proba = lgbc.predict_proba(X_test)[:, 1]

    if save:
        joblib.dump(lgbc, "../trained_models/lgbc_last.pkl")

    print("-----FINISHED LightGBM Classifier-----")

    return y_pred_proba
