import data_loader as dl
import pandas as pd
import numpy as np

TICKER = '^GSPC'
TARGET_WINDOW = 5

START = '2015-01-01'
END = '2026-01-01'
TIMEFRAME = '1d'

LOOKBACK_WINDOW = 14

def preprocess_data():
    story = dl.load(ticker=TICKER, start=START, end=END, timeframe=TIMEFRAME)

    data = pd.DataFrame()
    data['log_return'] = np.log(story['Close']).diff()

    for i in np.arange(1, LOOKBACK_WINDOW + 1):
        data[f'log_ret_lag_{i}'] = data['log_return'].shift(i)

    data['log_HL'] = np.log(story['High'] / story['Low'])
    data['log_CO'] = np.log(story['Close'] / story['Open'])

    data['log_volume'] = np.log(story['Volume'] + 1) # +1 to fix log(0)

    data['target'] = data['log_return'].rolling(window=TARGET_WINDOW).std().shift(-TARGET_WINDOW)

    data = data.dropna()
    return data