import data_loader as dl
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


import rolling_garch as rg

TICKER = '^GSPC'
TARGET_WINDOW = 5

START = '2015-01-01'
END = '2026-01-01'
TIMEFRAME = '1d'

LOOKBACK_WINDOW = 10
LOAD_EXISTING_DF = False
USE_GARCH = False

def preprocess_data() -> pd.DataFrame:
    if LOAD_EXISTING_DF:
        data = pd.read_csv('data/26-04-2024.csv', index_col=0)
        if not USE_GARCH and 'garch' in data.columns:
            data = data.drop(columns=['garch'])
        return data

    story = dl.load(ticker=TICKER, start=START, end=END, timeframe=TIMEFRAME)
    vix = dl.load(ticker='^VIX', start=START, end=END, timeframe=TIMEFRAME)

    data = pd.DataFrame()
    data['log_return'] = np.log(story['Close']).diff()
    data = data.dropna(subset=['log_return'])

    log_HL = np.log(story['High'] / story['Low']).reindex(data.index)
    log_CO = np.log(story['Close'] / story['Open']).reindex(data.index)

    data['log_volume'] = np.log(story['Volume'] + 1) # +1 to fix log(0)
    data['log_volume_diff'] = data['log_volume'].diff()

    data['log_ret_rolling_z'] = rolling_z_score(data['log_return'], LOOKBACK_WINDOW)

    data['garman-klass'] = np.sqrt(np.maximum( ((1 / 2) * log_HL**2 - (2*np.log(2) - 1) * log_CO**2), 0)) * 100 # possible negative values under sqrt
    data['gk_diff'] = data['garman-klass'].diff()
    data['gk_diff2'] = data['gk_diff'].diff()

    data['negative_return'] = np.minimum(data['log_return'], 0)
    data['positive_return'] = np.maximum(data['log_return'], 0)

    # data['day_of_week'] = data.index.dayofweek + 1
    data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 5)
    data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 5)

    data['log_vix'] = np.log(vix['Close']).reindex(data.index)
    data['log_vix_return'] = data['log_vix'].diff()

    data['gk_ma5'] = data['garman-klass'].rolling(window=5).mean()
    data['gk_ma21'] = data['garman-klass'].rolling(window=21).mean()

    overnight_log_ret = (np.log(story['Open'] / story['Close'].shift(1)) * 100)**2
    overnight_log_ret = overnight_log_ret.reindex(data.index).squeeze()

    # data['gk_adjusted'] = np.sqrt( data['garman-klass']**2 + overnight_log_ret )
    # data['target'] = data['gk_adjusted'].diff().shift(-1)
    # data['target'] = data['log_return'].rolling(window=TARGET_WINDOW).std().shift(-TARGET_WINDOW) * 100
    # data['target'] = data['parkinson'].shift(-1)
    # data['target'] = np.abs(data['log_return'].shift(-1) * 100)

    if USE_GARCH:
        data['garch'] = rg.rolling_garch(data['log_return'] * 100, 250)

    data['target'] = np.log(data['garman-klass']).diff().shift(-1)

    data = data.dropna()
    return data


def rolling_z_score(series, w):
    roll = series.rolling(window=w)
    return (series - roll.mean()) / roll.std()


def main():
    print('Saving .csv')
    data = preprocess_data()
    data.to_csv('data/26-04-2024.csv')
    data = data.loc['2019-01-01':]
    fig, ax = plt.subplots(figsize=(64,24))
    plt.plot(data['target'])
    plt.plot(data['garch'])
    fig.text(0.1, 0.94,f'GARCH прогноз: R^2: {r2_score(data['target'], data['garch']):.4f}')
    plt.show()



if __name__ == '__main__':
    main()