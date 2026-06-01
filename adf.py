import data_preprocessor as dp

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

data = dp.preprocess_data()

diff_series = data['target'].diff().dropna()

plt.plot(diff_series)
plt.show()
result = adfuller(diff_series)

print(f'ADF-статистика: {result[0]:.4f}, p-value: {result[1]:.4f}')