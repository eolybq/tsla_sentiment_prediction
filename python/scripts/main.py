import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_curve
import joblib
import glob
import os

from python.scripts.services.walk_forward import walk_forward_test
from python.scripts.services.evaluation import evaluate_class, evaluate_regg


# TODO
# mozna GD LR, XGBoost, LightGBM - overfitting?
# regularizace L1 / L2 u linear models



df = pd.read_csv('python/cleandata/processed_data.csv')


# TIMEFRAME PRO VYPOCET LOG RETURN
TIMEFRAME = 5
"""
 = 1 → denní return
 = 5 → týdenní return
 = 20 → měsíční return
 = 60 → kvartální return
"""

features = [
    # Unused
    # sma_20, sma_50, stochrsi, macd_signal, bb_up, bb_dn, g_trends

    # Log return vstupuje jen v lag -> autoregresni feature
    'log_return',
    'vix',
    'bull_bear_spread_surv',
    'volume',
    'ema_20',
    'basic_volatility', 'atr','macd', 'obv', 'rsi', 'adx',
    'sentiment_neutral', 'sentiment_positive', 'sentiment_negative', 'sentiment_none',
]
features_lin_models = features.copy()
features_lin_models.remove('sentiment_none')

# log return
df["log_return"] = np.log(df['adjusted'] / df['adjusted'].shift(TIMEFRAME))
df.dropna(inplace=True)


# def timeframe_agg(all_features, df):
#     if TIMEFRAME == 1:
#         return df
#
#     df_out = df[list(all_features.keys())].rolling(window=TIMEFRAME).agg(all_features).dropna()
#     return df_out

# vytvoreni umele sekvence lagu -> kazdy radek obsahuje Pocet f * Pocet lags
def create_lags(lags, all_features, df):
    X = pd.DataFrame()

    for l in range(1, lags + 1):
        for col in all_features:
            X[col + f'_lag{l}'] = df[col].shift(l)

    X.dropna(inplace=True)
    return X

lags = 10

# df_final = timeframe_agg(features, df)

X = create_lags(lags, features, df)
# Vynechani referencni urovne pro kategorie sentiment feature pro linearni modely
X_linear = create_lags(lags, features_lin_models, df)

features_names = X.columns


y = df["log_return"]
y_clf = (y > 0.005).astype(int)
# Omezeni radku dle hodnoty LAG
y, y_clf = y[lags:], y_clf[lags:]

# X, y, y_clf = np.array(X), np.array(y), np.array(y_clf)


gd_learning_rate = 0.01
gd_epochs = 10_000


window = 2630


# -----Main walk forward loop-----
walk_forward_test(
    TIMEFRAME,
    X,
    X_linear,
    y,
    y_clf,
    window,
    gd_learning_rate,
    gd_epochs,
)


# LOAD TRAINED MODELS AND PREDICTIONS:
loaded_models = {}
files = glob.glob("python/trained_models/*.pkl")


for f in files:
    base = os.path.basename(f)
    model_name = base.split("_")[0]
    loaded_models[model_name] = joblib.load(f)

preds_df = pd.read_csv("python/trained_models/predictions.csv", index_col=0)



# -----Evaluate-----
print("-----EVALUATION RESULTS-----")
# Posilat model name nejak aby hezky v grafu ale zaroven aby se daly soubory ulozit hezky

y_test = np.array(y[window:])
y_clf_test = y_clf[window:]


# Threshold maximalizujici F1
def find_optimal_threshold(y_true, y_proba):
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y_true, y_proba > t, average='weighted') for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]

    # fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # j_scores = tpr - fpr
    # best_thresh = thresholds[np.argmax(j_scores)]

    return best_thresh


# GD Linear Regression
print("---Gradient Descent Linear Regression---")
gd_lr_y_pred = preds_df['GD LinearRegression'].values
evaluate_regg(y_test, gd_lr_y_pred, model_name="GDLinearRegression")

# Decision Tree Reggressor
print("---Decision Tree Regressor---")
dtc_y_pred = preds_df['DecisionTreeRegressor'].values
evaluate_regg(y_test, dtc_y_pred, model_name="DecisionTreeRegressor", model=loaded_models["dtr"], features_names=features_names)

# Random Forest Reggressor
print("---Random Forest Regressor---")
rfr_y_pred = preds_df['RandomForestRegressor'].values
evaluate_regg(y_test, rfr_y_pred, model_name="RandomForestRegressor", model=loaded_models["rfr"], features_names=features_names)

# XGBoost Reggressor
print("---XGBoost Regressor---")
xgbr_y_pred = preds_df['XGBoostRegressor'].values
evaluate_regg(y_test, xgbr_y_pred, model_name="XGBoostRegressor", model=loaded_models["xgbr"], features_names=features_names)

# LightGBM Reggressor
print("---LightGBM Regressor---")
lgbr_y_pred = preds_df['LightGBMRegressor'].values
evaluate_regg(y_test, lgbr_y_pred, model_name="LightGBMRegressor", model=loaded_models["lgbr"], features_names=features_names)




# GD Logistic Regression
print("---Gradient Descent Logistic Regression---")
gd_logit_y_pred_proba = preds_df['GD LogisticRegression'].values
gd_logit_best_thresh = find_optimal_threshold(y_clf_test, gd_logit_y_pred_proba)
gd_logit_y_pred_class = (gd_logit_y_pred_proba >= gd_logit_best_thresh).astype(int)
evaluate_class(y_clf_test, gd_logit_y_pred_class, gd_logit_y_pred_proba, model_name="GDLogisticRegression")

# Decision Tree Classifier
print("---Decision Tree Classifier---")
dtc_y_pred_proba = preds_df['DecisionTreeClassifier'].values
dtc_best_thresh = find_optimal_threshold(y_clf_test, dtc_y_pred_proba)
dtc_y_pred_class = (dtc_y_pred_proba >= dtc_best_thresh).astype(int)
evaluate_class(y_clf_test, dtc_y_pred_class, dtc_y_pred_proba, model_name="DecisionTreeClassifier", model=loaded_models["dtc"], features_names=features_names)

# Random Forest Classifier
print("---Random Forest Classifier---")
rfc_y_pred_proba = preds_df['RandomForestClassifier'].values
rfc_best_thresh = find_optimal_threshold(y_clf_test, rfc_y_pred_proba)
rfc_y_pred_class = (rfc_y_pred_proba >= rfc_best_thresh).astype(int)
evaluate_class(y_clf_test, rfc_y_pred_class, rfc_y_pred_proba, model_name="RandomForestClassifier", model=loaded_models["rfc"], features_names=features_names)

# XGBoost Classifier
print("---XGBoost Classifier---")
xgbc_y_pred_proba = preds_df['XGBoostClassifier'].values
xgbc_best_thresh = find_optimal_threshold(y_clf_test, xgbc_y_pred_proba)
xgbc_y_pred_class = (xgbc_y_pred_proba >= xgbc_best_thresh).astype(int)
evaluate_class(y_clf_test, xgbc_y_pred_class, xgbc_y_pred_proba, model_name="XGBoostClassifier", model=loaded_models["xgbc"], features_names=features_names)

# LightGBM Classifier
print("---LightGBM Classifier---")
lgbc_y_pred_proba = preds_df['LightGBMClassifier'].values
lgbc_best_thresh = find_optimal_threshold(y_clf_test, lgbc_y_pred_proba)
lgbc_y_pred_class = (lgbc_y_pred_proba >= lgbc_best_thresh).astype(int)
evaluate_class(y_clf_test, lgbc_y_pred_class, lgbc_y_pred_proba, model_name="LightGBMClassifier", model=loaded_models["lgbc"], features_names=features_names)