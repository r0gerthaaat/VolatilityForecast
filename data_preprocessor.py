import data_loader as dl
import pandas as pd
import numpy as np

import rolling_garch as rg

TICKER = '^GSPC'
TARGET_WINDOW = 5

START = '2015-01-01'
END = '2026-01-01'
TIMEFRAME = '1d'

LOOKBACK_WINDOW = 10
LOAD_EXISTING_DF = True
USE_GARCH = True

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
    data['gk_ema_20'] = data['garman-klass'].ewm(span=20, adjust=False).mean()

    data['negative_return'] = np.minimum(data['log_return'], 0)
    data['positive_return'] = np.maximum(data['log_return'], 0)

    # data['day_of_week'] = data.index.dayofweek + 1
    data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 5)
    data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 5)

    data['log_vix'] = np.log(vix['Close']).reindex(data.index)
    data['log_vix_return'] = data['log_vix'].diff()

    data['gk-ma5'] = data['garman-klass'].rolling(window=5).mean()
    data['gk-ma21'] = data['garman-klass'].rolling(window=21).mean()

    #data['target'] = data['log_return'].rolling(window=TARGET_WINDOW).std().shift(-TARGET_WINDOW) * 100
    data['target'] = data['garman-klass'].shift(-1)
    #data['target'] = data['parkinson'].shift(-1)

    if USE_GARCH:
        data['garch'] = rg.rolling_garch(data['log_return'] * 100, 250)

    data = data.dropna()
    return data


def rolling_z_score(series, w):
    roll = series.rolling(window=w)
    return (series - roll.mean()) / roll.std()


def main():
    data = preprocess_data()
    data.to_csv('data/26-04-2024.csv')


if __name__ == '__main__':
    main()