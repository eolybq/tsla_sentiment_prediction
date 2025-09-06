import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cleandata/processed_data.csv')


fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# První graf
cols1 = ['adjusted', 'volume', 'vix', 'g_trends', 'bull_bear_spread_surv']
df_norm1 = df[cols1].apply(lambda x: (x - x.mean()) / x.std())
for col in cols1:
    axs[0].plot(df['date'], df_norm1[col], label=col)
axs[0].legend()
axs[0].set_title('Porovnání časových řad (skupina 1)')
axs[0].set_ylabel('Standardizovaná hodnota')

# Druhý graf
cols2 = ['adjusted', 'sma_20', 'sma_50', 'ema_20', 'basic_volatility', 'atr', 'macd', 'macd_signal', 'bb_up', 'bb_dn', 'obv', 'adx']
df_norm2 = df[cols2].apply(lambda x: (x - x.mean()) / x.std())
for col in cols2:
    axs[1].plot(df['date'], df_norm2[col], label=col)
axs[1].legend()
axs[1].set_title('Porovnání časových řad (skupina 2)')
axs[1].set_xlabel('Datum')
axs[1].set_ylabel('Standardizovaná hodnota')

plt.tight_layout()
plt.show()
