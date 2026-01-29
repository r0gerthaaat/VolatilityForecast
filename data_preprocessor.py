import data_loader as dl
import pandas as pd
import numpy as np

TICKER = '^GSPC'
TARGET_WINDOW = 5

story = dl.load(ticker=TICKER, start='2015-01-01', end='2026-01-01', timeframe='1d')

data = pd.DataFrame()
data['log_return'] = np.log(story['Close']).diff()

data['log_HL'] = np.log(story['High'] / story['Low'])
data['log_CO'] = np.log(story['Close'] / story['Open'])

data['log_volume'] = np.log(story['Volume'])

data['target'] = data['log_return'].rolling(window=TARGET_WINDOW).std().shift(-TARGET_WINDOW)

data = data.dropna()

print(data.tail(10))