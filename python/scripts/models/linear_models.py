import numpy as np


# Funkce na standartizace jen podle train dat (kazde window) -> dle stejnych mean a sd pak standartizace test dat
def standartize_features(X_train, X_test):
    X_train_numeric = X_train.select_dtypes(include=[np.number])

    mean = np.mean(X_train_numeric, axis=0)
    std = np.std(X_train_numeric, axis=0)

    X_train_standarized = (X_train_numeric - mean) / std
    X_test_standarized = (X_test - mean) / std

    return X_train_standarized, X_test_standarized



# -----GRADIENT DESCENT LINEAR REGRESSION------
def train_gd_lr(X_train, X_test, y_train, l_rate=0.01, epochs=10_000):
    print("-----TRAIN GD Linear Regression-----")

    X_train, X_test = standartize_features(X_train, X_test)


    weight = np.zeros(X_train.shape[1])
    bias = 0

    # mse_history = []

    for epoch in range(epochs):
        y_pred = np.dot(X_train, weight) + bias

        error = y_pred - y_train

        weight_grad = (2 / X_train.shape[0]) * X_train.T.dot(error)
        bias_grad = (2 / X_train.shape[0]) * np.sum(error)

        weight -= l_rate * weight_grad
        bias -= l_rate * bias_grad

    # NOTE "train analysis" pro puvodni ladění modelu -> do rolling window implementace nepotrebne

        # mse = (1 / X_train.shape[0]) * np.sum(np.square(error))
        # mse_history.append(mse)

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, MSE: {mse:.6f}")

    # plt.plot(y_train.values, label='Skutečný log-return')
    # plt.plot(y_pred, label='Predikce train')
    # plt.legend()
    # plt.show()

    # # R^2
    # sse_train = np.sum((y_train.values - y_pred)**2)
    # sst_train = np.sum((y_train.values - y_train.mean())**2)
    #
    # r2_train = 1 - sse_train / sst_train
    # print(f"Train R^2: {r2_train:.4f}")


    y_pred_test = np.dot(X_test, weight) + bias

    print("-----FINISHED GD Linear Regression-----")
    return y_pred_test



# -----LOGIT------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_gd_logit(X_train, X_test, y_train,l_rate=0.01, epochs=10_000):
    print("-----TRAIN GD Logistic Regression-----")

    X_train, X_test = standartize_features(X_train, X_test)


    weight = np.zeros(X_train.shape[1])
    bias = 0


    # loss_history = []

    for epoch in range(epochs):
        z = np.dot(X_train, weight) + bias
        y_pred = sigmoid(z)

        # Binary cross-entropy loss
        # loss = -np.mean(y_train * np.log(y_pred + 1e-8) + (1 - y_train) * np.log(1 - y_pred + 1e-8))
        # loss_history.append(loss)

        # Gradient
        error = y_pred - y_train
        weight_grad = np.dot(X_train.T, error) / X_train.shape[0]
        bias_grad = np.sum(error) / X_train.shape[0]

        weight -= l_rate * weight_grad
        bias -= l_rate * bias_grad

        # if epoch % 1000 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss:.6f}")


    # Predikce na testu
    y_pred_proba = sigmoid(np.dot(X_test, weight) + bias)

    print("-----FINISHED GD Logistic Regression-----")
    return y_pred_proba