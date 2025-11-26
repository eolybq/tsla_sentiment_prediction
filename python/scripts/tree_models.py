import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
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




# -----DECISION TREE CLASS------
print("-----Decision Tree Classifier-----")

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

    dtc = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    dtc.fit(X[start:i, ], y_clf[start:i])

    test = (X[i]).reshape(1, -1)

    prediction = dtc.predict_proba(test)[0][1]
    y_pred_test.append(prediction)

    y_clf_history.append(y_clf[i])
    print()
    print(f"Prediction: {prediction:.4f}, Actual: {y_clf[i]}")


dot_data = export_graphviz(dtc, out_file=None,
                           feature_names=features_names,
                           class_names=['0','1'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("../plots_tabs/decision_tree_class", cleanup=True)
graph.view()


# Feature importance
dtc_importances = pd.Series(dtc.feature_importances_, index=features_names)

# jen použité feature
dtc_importances_used = dtc_importances[dtc_importances > 0].sort_values(ascending=False)

dtc_importances_used.plot(kind='bar', figsize=(10,10))
plt.title("DTC Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.show()



y_pred_test = np.array(y_pred_test)
y_clf_history = np.array(y_clf_history)

threshold = 0.42
y_pred_class = (y_pred_test > threshold).astype(int)


acc = accuracy_score(y_clf_history, y_pred_class)
log_l = log_loss(y_clf_history, y_pred_class)
roc_auc = roc_auc_score(y_clf_history, y_pred_class)

metrics_df = pd.DataFrame({
     "Model": ["DecisionTreeClassifier"],
     "Accuracy": [acc],
     "Log Loss": [log_l],
     "ROC AUC": [roc_auc]
})

metrics_df.to_excel("../plots_tabs/dtc_metrics.xlsx", index=False)

print(f"Test accuracy: {acc:.4f}")
print(f"Log loss: {log_l:.6f}")
print(f"Roc auc: {roc_auc:.6f}")

print(classification_report(y_clf_history, y_pred_class))

cm = confusion_matrix(y_clf_history, y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("DTC Confusion Matrix")
plt.savefig("../plots_tabs/dtc_conf_m.png", dpi=300, bbox_inches='tight')
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
plt.title("Decision Tree Classifier")
plt.xlabel("")
# plt.savefig("../plots_tabs/gd_log.png", dpi=300, bbox_inches='tight')
plt.show()






# -----DECISION TREE REGG------
print("-----Decision Tree Regressor-----")

lags = 10
window = 2630

X = create_lags(lags)
features_names = X.columns

y = y[lags:]

X, y = np.array(X), np.array(y)


y_pred_test = []
y_history = []

for i in tqdm(range(window, len(X)), desc="Training"):
    start = i - window

    dtr = DecisionTreeRegressor(max_depth=5, random_state=42)
    dtr.fit(X[start:i, ], y[start:i])

    test = (X[i]).reshape(1, -1)

    prediction = dtr.predict(test)[0]
    y_pred_test.append(prediction)

    y_history.append(y[i])
    print()
    print(f"Prediction: {prediction}, Actual: {y[i]}")


dot_data = export_graphviz(dtr, out_file=None,
                           feature_names=features_names,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("../plots_tabs/decision_tree_reg", cleanup=True)
graph.view()


# Feature importance
dtr_importances = pd.Series(dtr.feature_importances_, index=features_names)

# jen použité feature
dtr_importances_used = dtr_importances[dtr_importances > 0].sort_values(ascending=False)

dtr_importances_used.plot(kind='bar', figsize=(10,10))
plt.title("DTR Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.show()



y_pred_test = np.array(y_pred_test)
y_history = np.array(y_history)



test_mse = mean_squared_error(y_history, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_history, y_pred_test)

# naive forecast: y_{t-1}
naive_pred = y_history[:-1]
naive_mse = mean_squared_error(y_history[1:], naive_pred)
naive_rmse = np.sqrt(naive_mse)
naive_mae = mean_absolute_error(y_history[1:], naive_pred)

test_mase = test_mae / naive_mae


print(f"Test MSE: {test_mse:.6f}")
print(f"Real: {y_history[-1]}, Predikce: {y_pred_test[-1]}")


# R^2
sse_test = np.sum((y_history - y_pred_test)**2)
sst_test = np.sum((y_history - y_history.mean())**2)

r2_test = 1 - sse_test / sst_test
print(f"Test R^2: {r2_test:.4f}")

metrics_df = pd.DataFrame({
    "Model": ["DecisionTreeRegressor", "Naive"],
    "MSE": [test_mse, naive_mse],
    "RMSE": [test_rmse, naive_rmse],
    "MAE": [test_mae, naive_mae],
    "MASE": [test_mase, 1.0000],
    "R^2": [r2_test, ""]
})
metrics_df.to_excel("../plots_tabs/dtr_metrics.xlsx", index=False)


custom_colors = {
    'y_pred': '#1717c1',
    'y_test': '#4D4D4D',
}

y_test_df = pd.DataFrame({
    'log_return': y_history,
    'variable': 'y_test'
}).reset_index()

y_pred_df = pd.DataFrame({
    'log_return': y_pred_test,
    'variable': 'y_pred'
}).reset_index()

test_pred_df = pd.concat([y_test_df, y_pred_df], ignore_index=True, axis=0)

sns.lineplot(data=test_pred_df, x="index", y='log_return', hue='variable', palette=custom_colors)
plt.xlabel('')
plt.ylabel('Log return')
plt.title("Decision Tree Regressor")
plt.savefig("../plots_tabs/dtr.png", dpi=300, bbox_inches='tight')
plt.show()