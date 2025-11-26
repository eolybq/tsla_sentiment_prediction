import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance, plot_tree, to_graphviz
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm

df = pd.read_csv('../cleandata/processed_data.csv')

features = [
    'vix', 'sentiment_neutral', 'sentiment_positive', 'sentiment_negative', "sentiment_none", 'g_trends',
    'bull_bear_spread_surv', 'volume', 'sma_20',
    'sma_50', 'ema_20', 'basic_volatility', 'atr', 'rsi', 'macd',
    'macd_signal', 'bb_up', 'bb_dn', 'obv', 'stochrsi', 'adx', "log_return"
]

df["log_return"] = np.log(df['adjusted'] / df['adjusted'].shift(1))
df.dropna(inplace=True)

y = df["log_return"]
y_clf = (y > 0.005).astype(int)


# vytvoreni umele sekvence lagu -> kazdy radek obsahuje Pocet f * Pocet lags
def create_lags(lags):
    X = pd.DataFrame()

    for l in range(1, lags + 1):
        for col in features:
            X[col + f'_lag{l}'] = df[col].shift(l)
    X.dropna(inplace=True)
    return X


# XGBOOST CLASS
print("-----XGBoost Classifier-----")

lags = 10
window = 2630

X = create_lags(lags)
features_names = X.columns

y_clf = y_clf[lags:]

X, y_clf = np.array(X), np.array(y_clf)


y_pred_test = []
y_clf_history = []

for i in tqdm(range(window, len(X)), desc="Training"):
    start = i - window

    xgb_c = XGBClassifier(objective="binary:logistic", max_depth=5, n_estimators=500, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_c.fit(X[start:i, ], y_clf[start:i])

    test = (X[i]).reshape(1, -1)

    prediction = xgb_c.predict_proba(test)[0][1]
    y_pred_test.append(prediction)

    y_clf_history.append(y_clf[i])
    print()
    print(f"Prediction: {prediction:.4f}, Actual: {y_clf[i]}")


graph = to_graphviz(xgb_c, num_trees=0)
graph.render("../plots_tabs/xgb_tree_class")  # uloží do xgb_tree.pdf
graph.view()


plot_importance(xgb_c, importance_type='gain')
plt.show()

plot_importance(xgb_c)
plt.show()



y_pred_test = np.array(y_pred_test)
y_clf_history = np.array(y_clf_history)

threshold = 0.42
y_pred_class = (y_pred_test > threshold).astype(int)


acc = accuracy_score(y_clf_history, y_pred_class)
log_l = log_loss(y_clf_history, y_pred_test)
roc_auc = roc_auc_score(y_clf_history, y_pred_test)

metrics_df = pd.DataFrame({
     "Model": ["XGBoostClassifier"],
     "Accuracy": [acc],
     "Log Loss": [log_l],
     "ROC AUC": [roc_auc]
})

metrics_df.to_excel("../plots_tabs/xgbc_metrics.xlsx", index=False)

print(f"Test accuracy: {acc:.4f}")
print(f"Log loss: {log_l:.6f}")
print(f"Roc auc: {roc_auc:.6f}")

print(classification_report(y_clf_history, y_pred_class))

cm = confusion_matrix(y_clf_history, y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGB Classifier Confusion Matrix")
plt.savefig("../plots_tabs/xgbc_conf_m.png", dpi=300, bbox_inches='tight')
plt.show()

custom_colors = {
     'Actual': '#4D4D4D',
     'Predicted': '#1717c1'
}

df_plot = pd.DataFrame({
     "Actual": y_clf_history,
     "Predicted": y_pred_class
})

sns.countplot(data=pd.melt(df_plot), x="value", hue="variable", palette=custom_colors)
plt.title("XGBoost Classifier")
plt.xlabel("")
# plt.savefig("../plots_tabs/gd_log.png", dpi=300, bbox_inches='tight')
plt.show()
