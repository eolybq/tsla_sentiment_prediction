import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('../cleandata/processed_data.csv')

# Stacionarita
# PCA?
# Transformace
# Zpozdeni


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

### GRADIENT DESCENT LINEAR REGRESSION
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

    mse = (1 / X_train.shape[0]) * np.sum(np.square(y_pred - y_train))
    mse_history.append(mse)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.6f}", weight)


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

print(f"Test MSE: {test_mse:.6f}")
print(f"Real: {y_test.values[-1]}, Predikce: {y_pred_test[-1]}")


# R^2
sse_test = np.sum((y_test.values - y_pred_test)**2)
sst_test = np.sum((y_test.values - y_test.mean())**2)

r2_test = 1 - sse_test / sst_test
print(f"Test R^2: {r2_test:.4f}")


plt.plot(y_test.values, label='Skutečný log-return')
plt.plot(y_pred_test, label='Predikce test')
plt.legend()
plt.show()




### LOGIT
y_train_d, y_test_d = (y_train > 0.005).astype(int), (y_test > 0.005).astype(int)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weight = np.zeros(X_train.shape[1])
bias = 0
rate = 0.008
epochs = 100000
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

print(f"Test accuracy: {accuracy_score(y_test_d, y_pred_class)}")

cm = confusion_matrix(y_test_d, y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test_d, y_pred_class))

print(f"Log loss: {log_loss(y_test_d, y_pred_test)}")

print(f"Roc auc: {roc_auc_score(y_test_d, y_pred_test)}")