import pandas as pd
import numpy as np
import statsmodels.api as sm

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
    'vix', 'sentiment', 'g_trends', 'bullish_surv', 'neutral_surv',
    'bearish_surv', 'bull_bear_spread_surv', 'volume', 'sma_20',
    'sma_50', 'ema_20', 'basic_volatility', 'atr', 'rsi', 'macd',
    'macd_signal', 'bb_up', 'bb_dn', 'obv', 'stochrsi', 'adx'
]

for col in features:
    df[col + '_lag1'] = df[col].shift(1)


df['log_return'] = np.insert(np.diff(np.log(df['adjusted'])), 0, np.nan)

# Připrav X a y
X = df[[col + '_lag1' for col in features]]
X = pd.get_dummies(X, columns=['sentiment_lag1'], drop_first=True)
y = df['log_return']

# Odstraň řádky s NaN (vzniklé posunem)
df = df.dropna(subset=[col + '_lag1' for col in features] + ['adjusted'])
X = X.loc[df.index]
y = y.loc[df.index]

X = X.astype(float)

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
