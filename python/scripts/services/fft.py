import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('python/cleandata/processed_data.csv')
x = df['adjusted'].values

# Fourierova transformace
X = np.fft.fft(x)
freq = np.fft.fftfreq(len(x))
print(len(X))
# Spektrum (absolutní hodnota)
plt.figure(figsize=(12,4))
plt.plot(freq, np.abs(X))
plt.title('Frekvenční spektrum')
plt.xlabel('Frekvence')
plt.ylabel('Amplituda')
plt.show()

# Filtrace: ponecháme jen několik největších frekvencí
X_filtered = np.copy(X)
X_filtered[np.abs(X) < np.percentile(np.abs(X), 99)] = 0  # např. ponecháme jen 1 % nejsilnějších složek

# Rekonstrukce signálu
x_reconstructed = np.fft.ifft(X_filtered)

plt.figure(figsize=(12,4))
plt.plot(x, label='Původní')
plt.plot(x_reconstructed.real, label='Rekonstruovaný (hlavní cykly)')
plt.legend()
plt.title('Porovnání původní a filtrované řady')
plt.show()