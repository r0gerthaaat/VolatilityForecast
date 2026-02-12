import data_loader as dl
import pandas as pd
import numpy as np

TICKER = '^GSPC'
TARGET_WINDOW = 5

START = '2015-01-01'
END = '2026-01-01'
TIMEFRAME = '1d'

LOOKBACK_WINDOW = 10

def preprocess_data():
    story = dl.load(ticker=TICKER, start=START, end=END, timeframe=TIMEFRAME)

    data = pd.DataFrame()
    data['log_return'] = np.log(story['Close']).diff()

    log_HL = np.log(story['High'] / story['Low'])
    log_CO = np.log(story['Close'] / story['Open'])

    data['log_volume'] = np.log(story['Volume'] + 1) # +1 to fix log(0)

    data['log_ret_rolling_z'] = rolling_z_score(data['log_return'], LOOKBACK_WINDOW)

    data['parkinson'] = np.sqrt( (1 / (4 * np.log(2))) * (log_HL ** 2) )
    data['parkinson_rolling'] = data['parkinson'].rolling(window=TARGET_WINDOW).mean()

    data['garman-klass'] = np.sqrt(np.maximum( ((1 / 2) * log_HL**2 - (2*np.log(2) - 1) * log_CO**2), 0)) # possible negative values under sqrt
    data['garman-klass_rolling'] = data['garman-klass'].rolling(window=TARGET_WINDOW).mean()

    data['day_of_week'] = data.index.dayofweek + 1

    data['target'] = data['log_return'].rolling(window=TARGET_WINDOW).std().shift(-TARGET_WINDOW)

    data = data.dropna()
    return data


def rolling_z_score(series, w):
    roll = series.rolling(window=w)
    return (series - roll.mean()) / roll.std()