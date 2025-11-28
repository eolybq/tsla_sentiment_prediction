import numpy as np
import pandas as pd
from tqdm import tqdm

from python.scripts.models import train_gd_lr, train_gd_logit, train_dtr, train_dtc, train_rfr, train_rfc, train_xgbr, train_xgbc, train_lgbr, train_lgbc

def walk_forward_test(
    X,
    y,
    y_clf,
    window,
    gd_learning_rate,
    gd_epochs,
):

    preds_df = pd.DataFrame(
        np.nan,
        index=range(window, len(X)),
        columns = [
            'GD LinearRegression',
            'DecisionTreeRegressor',
            'RandomForestRegressor',
            'XGBoostRegressor',
            'LightGBMRegressor',
            'GD LogisticRegression',
            'DecisionTreeClassifier',
            'RandomForestClassifier',
            'XGBoostClassifier',
            'LightGBMClassifier'
        ]
    )

    preds_df.to_csv("python/trained_models/predictions.csv", index=True)

    # -----Main walk forward loop-----
    for i in tqdm(range(window, len(X)), desc="Training"):
        start = i - window

        X_train = X.iloc[start:i, ]
        y_train = y.iloc[start:i]
        y_clf_train = y_clf.iloc[start:i]

        # X_test = (X[i]).reshape(1, -1)
        X_test = X.iloc[[i], :]


        # GD Models
        gd_lr_y_pred = train_gd_lr(X_train, X_test, y_train, gd_learning_rate, gd_epochs)
        preds_df.loc[i, 'GD LinearRegression'] = gd_lr_y_pred

        gd_logit_y_pred_proba = train_gd_logit(X_train, X_test, y_clf_train, gd_learning_rate, gd_epochs)
        preds_df.loc[i, 'GD LogisticRegression'] = gd_logit_y_pred_proba


        save_model_flag = (i == len(X) - 1)

        # Decision Trees
        dtr_y_pred = train_dtr(X_train, X_test, y_train, max_depth=5, save=save_model_flag)
        preds_df.loc[i, 'DecisionTreeRegressor'] = dtr_y_pred

        dtc_y_pred_proba = train_dtc(X_train, X_test, y_clf_train, max_depth=5, save=save_model_flag)
        preds_df.loc[i, 'DecisionTreeClassifier'] = dtc_y_pred_proba


        # Random Forests
        rfr_y_pred = train_rfr(X_train, X_test, y_train, n_estimators=500, max_depth=5, save=save_model_flag)
        preds_df.loc[i, 'RandomForestRegressor'] = rfr_y_pred

        rfc_y_pred_proba = train_rfc(X_train, X_test, y_clf_train, n_estimators=500, max_depth=5, save=save_model_flag)
        preds_df.loc[i, 'RandomForestClassifier'] = rfc_y_pred_proba


        # XGBoosts
        xgbr_y_pred = train_xgbr(X_train, X_test, y_train, n_estimators=500, max_depth=5, save=save_model_flag)
        preds_df.loc[i, 'XGBoostRegressor'] = xgbr_y_pred

        xgbc_y_pred_proba = train_xgbc(X_train, X_test, y_clf_train, n_estimators=500, max_depth=5, save=save_model_flag)
        preds_df.loc[i, 'XGBoostClassifier'] = xgbc_y_pred_proba


        # LightGBMs
        lgbr_y_pred = train_lgbr(X_train, X_test, y_train, n_estimators=500, max_depth=5, save=save_model_flag)
        preds_df.loc[i, 'LightGBMRegressor'] = lgbr_y_pred

        lgbc_y_pred_proba = train_lgbc(X_train, X_test, y_clf_train, n_estimators=500, max_depth=5, save=save_model_flag)
        preds_df.loc[i, 'LightGBMClassifier'] = lgbc_y_pred_proba


        # prubezne ukladnani predikci
        preds_df.to_csv("python/trained_models/predictions.csv", index=True)
