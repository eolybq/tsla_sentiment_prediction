import pandas as pd
import numpy as np
import joblib
import glob
from tqdm import tqdm

from models.linear_models import train_gd_lr, train_gd_logit
from models.tree_models import train_dtr, train_dtc
from models.forest_models import train_rfr, train_rfc
from models.xgboost_models import train_xgbr, train_xgbc
from models.lightgbm_models import train_lgbr, train_lgbc


df = pd.read_csv('../cleandata/processed_data.csv')

# log return
df["log_return"] = np.log(df['adjusted'] / df['adjusted'].shift(1))
df.dropna(inplace=True)


# PCA?

features = [
    'log_return', 'vix', 'sentiment_neutral', 'sentiment_positive', 'sentiment_negative', 'sentiment_none', 'g_trends',
    'bull_bear_spread_surv', 'volume', 'sma_20',
    'sma_50', 'ema_20', 'basic_volatility', 'atr', 'rsi', 'macd',
    'macd_signal', 'bb_up', 'bb_dn', 'obv', 'stochrsi', 'adx'
]

features_lin_models = features.copy()
features_lin_models.remove("sentiment_none")



# vytvoreni umele sekvence lagu -> kazdy radek obsahuje Pocet f * Pocet lags
def create_lags(lags):
    X = pd.DataFrame()

    for l in range(1, lags + 1):
        for col in features:
            X[col + f'_lag{l}'] = df[col].shift(l)

    X.dropna(inplace=True)
    return X

lags = 10

X = create_lags(lags)
features_names = X.columns

y = df["log_return"]
y_clf = (y > 0.005).astype(int)

# Omezeni radku dle hodnoty LAG
y, y_clf = y[lags:], y_clf[lags:]

# X, y, y_clf = np.array(X), np.array(y), np.array(y_clf)


pred_threshold = 0.5
learning_rate = 0.01
epochs = 10_000



window = 2630


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

preds_df.to_csv("../trained_models/predictions.csv", index=True)

# -----Main walk forward loop-----
for i in tqdm(range(window, len(X)), desc="Training"):
    start = i - window

    X_train = X.iloc[start:i, ]
    y_train = y.iloc[start:i]
    y_clf_train = y_clf.iloc[start:i]

    # X_test = (X[i]).reshape(1, -1)
    X_test = X.iloc[[i], :]


    # GD Models
    gd_lr_y_pred = train_gd_lr(X_train, X_test, y_train, learning_rate, epochs)
    preds_df.loc[i, 'GD LinearRegression'] = gd_lr_y_pred

    gd_logit_y_pred_class, gd_logit_y_pred_proba = train_gd_logit(X_train, X_test, y_clf_train, pred_threshold, learning_rate, epochs)
    preds_df.loc[i, 'GD LogisticRegression'] = gd_logit_y_pred_proba


    save_model_flag = (i == len(X) - 1)

    # Decision Trees
    dtr_y_pred = train_dtr(X_train, X_test, y_train, max_depth=5, save=save_model_flag)
    preds_df.loc[i, 'DecisionTreeRegressor'] = dtr_y_pred

    dtc_y_pred_class, dtc_y_pred_proba = train_dtc(X_train, X_test, y_clf_train, pred_threshold, max_depth=5, save=save_model_flag)
    preds_df.loc[i, 'DecisionTreeClassifier'] = dtc_y_pred_proba


    # Random Forests
    rfr_y_pred = train_rfr(X_train, X_test, y_train, n_estimators=500, max_depth=5, save=save_model_flag)
    preds_df.loc[i, 'RandomForestRegressor'] = rfr_y_pred

    rfc_y_pred_class, rfc_y_pred_proba = train_rfc(X_train, X_test, y_clf_train, pred_threshold, n_estimators=500, max_depth=5, save=save_model_flag)
    preds_df.loc[i, 'RandomForestClassifier'] = rfc_y_pred_proba


    # XGBoosts
    xgbr_y_pred = train_xgbr(X_train, X_test, y_train, n_estimators=500, max_depth=5, save=save_model_flag)
    preds_df.loc[i, 'XGBoostRegressor'] = xgbr_y_pred

    xgbc_y_pred_class, xgbc_y_pred_proba = train_xgbc(X_train, X_test, y_clf_train, pred_threshold, n_estimators=500, max_depth=5, save=save_model_flag)
    preds_df.loc[i, 'XGBoostClassifier'] = xgbc_y_pred_proba


    # LightGBMs
    lgbr_y_pred = train_lgbr(X_train, X_test, y_train, n_estimators=500, max_depth=5, save=save_model_flag)
    preds_df.loc[i, 'LightGBMRegressor'] = lgbr_y_pred

    lgbc_y_pred_class, lgbc_y_pred_proba = train_lgbc(X_train, X_test, y_clf_train, pred_threshold, n_estimators=500, max_depth=5, save=save_model_flag)
    preds_df.loc[i, 'LightGBMClassifier'] = lgbc_y_pred_proba


    # prubezne ukladnani predikci
    preds_df.to_csv("../trained_models/predictions.csv", index=True)




# LOAD TRAINED MODELS AND PREDICTIONS:
loaded_models = {}
files = glob.glob("*.pkl")

for f in files:
    model_name = f.split(".pkl")[0].split("_")[0]
    loaded_models[model_name] = joblib.load(f)

preds_df = pd.read_csv("../trained_models/predictions.csv", index_col=0)



# -----Evaluate-----
y_test = y[window:]
y_clf_test = y_clf[window:]