import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('cleandata/processed_data.csv')

# Stacionarita
# PCA?
# Transformace
# Zpozdeni

# Začni jednoduchým modelem (např. Random Forest, XGBoost, lineární regrese).
# Přidej zpožděné proměnné (lagy) – často výrazně zlepší predikci časových řad.
# Vyzkoušej sekvenční modely (LSTM/RNN), pokud máš dost dat a zkušeností.
# Vyhodnoť modely pomocí cross-validace a metrik jako RMSE, MAE.


features = [
    'vix', 'sentiment_neutral', 'sentiment_positive', 'sentiment_negative', 'g_trends',
    'bull_bear_spread_surv', 'volume', 'sma_20',
    'sma_50', 'ema_20', 'basic_volatility', 'atr', 'rsi', 'macd',
    'macd_signal', 'bb_up', 'bb_dn', 'obv', 'stochrsi', 'adx'
]



for col in features:
    df[col + '_lag1'] = df[col].shift(1)
    df[col + '_lag2'] = df[col].shift(2)

df['adjusted'] = np.log(df['adjusted'] / df['adjusted'].shift(1))
df = df.dropna().reset_index(drop=True)


X = df[[f + '_lag1' for f in features] + [f + '_lag2' for f in features]].copy()
X['adjusted_lag1'] = df['adjusted'].shift(1)
X = X.dropna()

y = df['adjusted']
y = y.iloc[1:]



fig, ax = plt.subplots(2, 1, figsize=(10,6))
# ACF
plot_acf(y, ax=ax[0], lags=40)
ax[0].set_title(f"ACF pro adjusted")
# PACF
plot_pacf(y, ax=ax[1], lags=40)
ax[1].set_title(f"PACF pro adjusted")
plt.tight_layout()
plt.show()



for column in X:
    if "sentiment" not in column:
        X[column] = (X[column] - X[column].mean()) / X[column].std()





X = np.array(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)



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




y_pred_test = np.dot(X_test, weight) + bias
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Test MSE: {test_mse:.6f}")
print(f"Real: {y_test.values[-1]}, Predikce: {y_pred_test[-1]}")

plt.plot(y_test.values, label='Skutečný log-return')
plt.plot(y_pred_test, label='Predikce test')
plt.legend()
plt.show()




### LOGIT
y = (y > 0.05).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)




def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weight = np.zeros(X_train.shape[1])
bias = 0
rate = 0.01
epochs = 10000
loss_history = []

for epoch in range(epochs):
    z = np.dot(X_train, weight) + bias
    y_pred = sigmoid(z)
    # Binary cross-entropy loss
    loss = -np.mean(y_train * np.log(y_pred + 1e-8) + (1 - y_train) * np.log(1 - y_pred + 1e-8))
    loss_history.append(loss)

    # Gradient
    error = y_pred - y_train
    weight_grad = np.dot(X_train.T, error) / X_train.shape[0]
    bias_grad = np.sum(error) / X_train.shape[0]

    weight -= rate * weight_grad
    bias -= rate * bias_grad

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Predikce na testu
y_pred_test = sigmoid(np.dot(X_test, weight) + bias)
y_pred_class = (y_pred_test > 0.5).astype(int)

from sklearn.metrics import accuracy_score
print("Test accuracy:", accuracy_score(y_test, y_pred_class))