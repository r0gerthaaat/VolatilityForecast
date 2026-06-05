from src.data import data_loader as dl
import pandas as pd
import numpy as np

from src.models import rolling_garch as rg

from pathlib import Path

TICKER = '^GSPC'
TARGET_WINDOW = 5

START = '2015-01-01'
END = '2026-01-01'
TIMEFRAME = '1d'

LOOKBACK_WINDOW = 10
LOAD_EXISTING_DF = True
USE_GARCH = True
USE_DIFFERENCES = False

current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent.parent

def preprocess_data() -> pd.DataFrame:
    file_diff = project_root / 'data' / 'target-differences.csv'
    file_orig = project_root / 'data' / 'target-original.csv'

    if LOAD_EXISTING_DF:
        if USE_DIFFERENCES and file_diff.exists():
            print(f'Loading existing data from {file_diff}')
            return pd.read_csv(file_diff, index_col=0, parse_dates=True)
        elif not USE_DIFFERENCES and file_orig.exists():
            print(f'Loading existing data from {file_orig}')
            return pd.read_csv(file_orig, index_col=0, parse_dates=True)

    print(f'Generating new data... Differences are target: {USE_DIFFERENCES}')

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

    data['garman-klass'] = np.sqrt(np.maximum( ((1 / 2) * log_HL**2 - (2*np.log(2) - 1) * log_CO**2), 0)) * 100
    data['gk_diff'] = data['garman-klass'].diff()
    data['gk_diff2'] = data['gk_diff'].diff()

    data['negative_return'] = np.minimum(data['log_return'], 0)
    data['positive_return'] = np.maximum(data['log_return'], 0)

    data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 5)
    data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 5)

    data['log_vix'] = np.log(vix['Close']).reindex(data.index)
    data['log_vix_return'] = data['log_vix'].diff()

    data['gk_ma5'] = data['garman-klass'].rolling(window=5).mean()
    data['gk_ma21'] = data['garman-klass'].rolling(window=21).mean()

    # overnight_log_ret = (np.log(story['Open'] / story['Close'].shift(1)) * 100)**2
    # overnight_log_ret = overnight_log_ret.reindex(data.index).squeeze()
    #
    # data['gk_adjusted'] = np.sqrt( data['garman-klass']**2 + overnight_log_ret )
    # data['target'] = data['gk_adjusted'].shift(-1)

    if USE_GARCH:
        data['garch'] = rg.rolling_garch(data['log_return'] * 100, 250)

    if USE_DIFFERENCES:
        data['target'] = data['garman-klass'].diff().shift(-1)
    else:
        data['target'] = data['garman-klass'].shift(-1)

    data = data.dropna()

    if USE_DIFFERENCES:
        data.to_csv(file_diff)
        print(f'Saved new data to {file_diff}')
    else:
        data.to_csv(file_orig)
        print(f'Saved new data to {file_orig}')

    return data


def rolling_z_score(series, w):
    roll = series.rolling(window=w)
    return (series - roll.mean()) / roll.std()


def get_CO_log_rets():
    file_orig = project_root / 'data' / 'target-original-with-log-open-to-close-returns.csv'
    data = pd.read_csv(file_orig, index_col=0, parse_dates=True)
    return data['co_log_ret']