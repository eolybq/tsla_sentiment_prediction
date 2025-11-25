import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
from tqdm import tqdm


df = pd.read_csv('../cleandata/processed_data.csv')

# TODO udealt feature selection - oddelat rsi a macd_signal a podobne blbosti
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


# ------RANDOM FOREST CLASS------
print("-----Random Forest Classifier-----")

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

    rfc = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=5, random_state=42, n_jobs=-1)
    rfc.fit(X[start:i, ], y_clf[start:i])

    test = (X[i]).reshape(1, -1)

    prediction = rfc.predict_proba(test)[0][1]
    y_pred_test.append(prediction)

    y_clf_history.append(y_clf[i])
    print()
    print(f"Prediction: {prediction:.4f}, Actual: {y_clf[i]}")


best = {
    "best_acc":0,
    "best_tree": None
}

for i, tree in enumerate(rfc.estimators_):
    y_pred_one = tree.predict(X[-1, ].reshape(1, -1))
    acc = y_clf[-1] - y_pred_one

    if acc > best["best_acc"]:
        best["best_acc"] = acc
        best["best_tree"] = tree

dot_data = export_graphviz(best["best_tree"], out_file=None,
                           feature_names=features_names,
                           class_names=['0','1'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("../plots_tabs/rfc_best_tree", cleanup=True)
graph.view()


# Feature importance
rfc_importances = pd.Series(rfc.feature_importances_, index=features_names)

# jen použité feature
rfc_importances_used = rfc_importances[rfc_importances > 0].sort_values(ascending=False)

rfc_importances_used.plot(kind='bar', figsize=(10,10))
plt.title("RFC Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.show()



y_pred_test = np.array(y_pred_test)
y_clf_history = np.array(y_clf_history)

threshold = 0.42
y_pred_class = (y_pred_test > threshold).astype(int)


acc = accuracy_score(y_clf_history, y_pred_class)
log_l = log_loss(y_clf_history, y_pred_test)
roc_auc = roc_auc_score(y_clf_history, y_pred_test)

metrics_df = pd.DataFrame({
     "Model": ["RandomForestClassifier"],
     "Accuracy": [acc],
     "Log Loss": [log_l],
     "ROC AUC": [roc_auc]
})

metrics_df.to_excel("../plots_tabs/rfc_metrics.xlsx", index=False)

print(f"Test accuracy: {acc:.4f}")
print(f"Log loss: {log_l:.6f}")
print(f"Roc auc: {roc_auc:.6f}")

print(classification_report(y_clf_history, y_pred_class))

cm = confusion_matrix(y_clf_history, y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("RFC Confusion Matrix")
plt.savefig("../plots_tabs/rfc_conf_m.png", dpi=300, bbox_inches='tight')
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
plt.title("Random Forest Classifier")
plt.xlabel("")
# plt.savefig("../plots_tabs/gd_log.png", dpi=300, bbox_inches='tight')
plt.show()


