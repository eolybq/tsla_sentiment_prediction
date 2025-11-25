import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('../cleandata/processed_data.csv')

# Stacionarita
# PCA?
# Transformace
# Zpozdeni


# TODO udealt feature selection - oddelat rsi a macd_signal a podobne blbosti
features = [
    'vix', 'sentiment_neutral', 'sentiment_positive', 'sentiment_negative', 'g_trends',
    'bull_bear_spread_surv', 'volume', 'sma_20',
    'sma_50', 'ema_20', 'basic_volatility', 'atr', 'rsi', 'macd',
    'macd_signal', 'bb_up', 'bb_dn', 'obv', 'stochrsi', 'adx'
]



X = pd.DataFrame()
for col in features:
    X[col + '_lag1'] = df[col].shift(1)
    X[col + '_lag2'] = df[col].shift(2)

# log return
y = np.log(df['adjusted'] / df['adjusted'].shift(1))

X['log_return_lag1'] = y.shift(1)

# Drop 2 prvnich radku kvuli max lag = 2 a zarovnani indexu
X = X.dropna().reset_index(drop=True)
y = y.iloc[2:].reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

# Standartizace jen podle train dat -> dle stejnych mean a sd pak standartizace test dat
for column in X_train.columns:
    if "sentiment" not in column:
        mean = X_train[column].mean()
        std = X_train[column].std()
        X_train[column] = (X_train[column] - mean) / std
        X_test[column] = (X_test[column] - mean) / std


X_train, X_test = np.array(X_train), np.array(X_test)




# -----GRADIENT DESCENT LINEAR REGRESSION------
weight = np.zeros(X.shape[1])
bias = 0
rate = 0.01
epochs = 20000
mse_history = []

for epoch in range(epochs):
    y_pred = np.dot(X_train, weight) + bias

    error = y_pred - y_train

    weight_grad = (2 / X_train.shape[0]) * X_train.T.dot(error)
    bias_grad = (2 / X_train.shape[0]) * np.sum(error)

    weight -= rate * weight_grad
    bias -= rate * bias_grad

    mse = (1 / X_train.shape[0]) * np.sum(np.square(error))
    mse_history.append(mse)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.6f}")

plt.plot(y_train.values, label='Skutečný log-return')
plt.plot(y_pred, label='Predikce train')
plt.legend()
plt.show()





# R^2
sse_train = np.sum((y_train.values - y_pred)**2)
sst_train = np.sum((y_train.values - y_train.mean())**2)

r2_train = 1 - sse_train / sst_train
print(f"Train R^2: {r2_train:.4f}")



y_pred_test = np.dot(X_test, weight) + bias
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)

# naive forecast: y_{t-1}
naive_pred = y_test.shift(1).dropna()
naive_mse = mean_squared_error(y_test.iloc[1:], naive_pred)
naive_rmse = np.sqrt(naive_mse)
naive_mae = mean_absolute_error(y_test.iloc[1:], naive_pred)

test_mase = test_mae / naive_mae


print(f"Test MSE: {test_mse:.6f}")
print(f"Real: {y_test.values[-1]}, Predikce: {y_pred_test[-1]}")


# R^2
sse_test = np.sum((y_test.values - y_pred_test)**2)
sst_test = np.sum((y_test.values - y_test.mean())**2)

r2_test = 1 - sse_test / sst_test
print(f"Test R^2: {r2_test:.4f}")

metrics_df = pd.DataFrame({
    "Model": ["GD LinearRegression", "Naive"],
    "MSE": [test_mse, naive_mse],
    "RMSE": [test_rmse, naive_rmse],
    "MAE": [test_mae, naive_mae],
    "MASE": [test_mase, 1.0000],
    "R^2": [r2_test, ""]
})
metrics_df.to_excel("../plots_tabs/gd_lr_metrics.xlsx", index=False)


custom_colors = {
    'y_pred': '#1717c1',
    'y_test': '#4D4D4D',
}

y_test_df = pd.DataFrame({
    'log_return': y_test.values,
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
plt.title("Gradient Descent LinearRegression")
plt.savefig("../plots_tabs/gd_lr.png", dpi=300, bbox_inches='tight')
plt.show()




# -----LOGIT------
y_train_d, y_test_d = (y_train > 0.005).astype(int), (y_test > 0.005).astype(int)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weight = np.zeros(X_train.shape[1])
bias = 0
rate = 0.008
epochs = 20000
loss_history = []

for epoch in range(epochs):
    z = np.dot(X_train, weight) + bias
    y_pred = sigmoid(z)
    # Binary cross-entropy loss
    loss = -np.mean(y_train_d * np.log(y_pred + 1e-8) + (1 - y_train_d) * np.log(1 - y_pred + 1e-8))
    loss_history.append(loss)

    # Gradient
    error = y_pred - y_train_d
    weight_grad = np.dot(X_train.T, error) / X_train.shape[0]
    bias_grad = np.sum(error) / X_train.shape[0]

    weight -= rate * weight_grad
    bias -= rate * bias_grad

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")


# Predikce na testu
y_pred_test = sigmoid(np.dot(X_test, weight) + bias)

threshold = 0.35
y_pred_class = (y_pred_test > threshold).astype(int)


# Metrics
# print(f"Trefeno 1: {np.sum((y_pred_class == y_test_d) & (y_pred_class == 1))} Trefeno 0: {np.sum((y_pred_class == y_test_d) & (y_pred_class == 0))}, Celkem: {len(y_pred_class)}")


acc = accuracy_score(y_test_d, y_pred_class)
log_l = log_loss(y_test_d, y_pred_test)
roc_auc = roc_auc_score(y_test_d, y_pred_test)

metrics_df = pd.DataFrame({
    "Model": ["GD LogisticRegression"],
    "Accuracy": [acc],
    "Log Loss": [log_l],
    "ROC AUC": [roc_auc]
})

metrics_df.to_excel("../plots_tabs/gd_log_metrics.xlsx", index=False)

print(f"Test accuracy: {acc:.4f}")
print(f"Log loss: {log_l:.6f}")
print(f"Roc auc: {roc_auc:.6f}")

print(classification_report(y_test_d, y_pred_class))

cm = confusion_matrix(y_test_d, y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("GD Logit Confusion Matrix")
plt.savefig("../plots_tabs/gd_log_conf_m.png", dpi=300, bbox_inches='tight')
plt.show()


custom_colors = {
    'Actual': '#4D4D4D',
    'Predicted': '#1717c1'
}

df_plot = pd.DataFrame({
    "Actual": y_test_d,
    "Predicted": y_pred_class
})

sns.countplot(data=pd.melt(df_plot), x="value", hue="variable", palette=custom_colors)
plt.title("Gradient Descent LogisticRegression")
plt.xlabel("")
# plt.savefig("../plots_tabs/gd_log.png", dpi=300, bbox_inches='tight')
plt.show()
