import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.nn import MSELoss, HuberLoss

import data_preprocessor as dp
from dataset import VolatilityDataset
from model import VolatilityLSTM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from qlike import QLIKELoss

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from arch import arch_model

import random
import os

from scipy import stats

import dieboldmariano as dm

BATCH_SIZE = 64

INPUT_SIZE = 11
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2

LEARNING_RATE = 0.001
EPOCHS = 23
LOSS_FUNCTION = HuberLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def main():
    seed_everything()

    # DATA PREPARING BLOCK
    dp.LOOKBACK_WINDOW = 21
    data = dp.preprocess_data()
    print(f'{len(data)} rows in the loaded data')

    INPUT_SIZE = len(data.columns) - 1

    train_data, test_data = train_test_split(data, test_size=0.1, shuffle=False)
    train_x = train_data.drop(columns=['target'])
    train_y = train_data[['target']]

    test_x = test_data.drop(columns=['target'])
    test_y = test_data[['target']]

    x_scaler: StandardScaler = StandardScaler().fit(train_x)
    # y_scaler: StandardScaler = StandardScaler().fit(train_y)

    train_x_scaled: np.ndarray = x_scaler.transform(train_x)
    train_y_scaled: np.ndarray = train_y.values#y_scaler.transform(train_y)

    test_x_scaled: np.ndarray = x_scaler.transform(test_x)
    test_y_scaled: np.ndarray = test_y.values#y_scaler.transform(test_y)

    train_ds = VolatilityDataset(train_x_scaled, train_y_scaled, dp.LOOKBACK_WINDOW)
    test_ds = VolatilityDataset(test_x_scaled, test_y_scaled, dp.LOOKBACK_WINDOW)

    train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # shuffle for pattern recognizing instead of chronologic remembering
    test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # MODEL BLOCK
    model = VolatilityLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    criterion = LOSS_FUNCTION
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    train_losses = []
    test_losses = []

    print('Start training...')

    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0

        predictions = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)

            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        # VALIDATION
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                out = model(x_batch)
                predictions.append(out)
                loss = criterion(out, y_batch)

                epoch_test_loss += loss.item()

            average_train_loss = epoch_train_loss / len(train_loader)
            average_test_loss = epoch_test_loss / len(test_loader)

            train_losses.append(average_train_loss)
            test_losses.append(average_test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], train loss: {average_train_loss}, test loss: {average_test_loss}')


    print(f'Minimum loss epoch: {np.argmin(test_losses) + 1}')

    # VISUALIZING
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Times New Roman'],
        'font.size': 14
    })
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.lineplot(train_losses, color='red', label='Навчання')
    sns.lineplot(test_losses, color='blue', label='Тестування')
    plt.title('')
    plt.ylabel('Функція втрат')
    plt.xlabel('Епоха')
    plt.legend()
    plt.grid()
    plt.show()

    test_predictions = []
    test_actuals = []
    all_attention_weights = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            out, attn_weights = model(x_batch, return_attention=True)

            out = out.cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy()
            attn_weights = attn_weights.cpu().detach().numpy()

            test_predictions.append(out)
            test_actuals.append(y_batch)
            all_attention_weights.append(attn_weights)


    test_predictions = np.vstack(test_predictions).flatten()
    test_actuals = np.vstack(test_actuals).flatten()

    raw_diffs_actuals = test_actuals.copy()  # saving diffs
    raw_diffs_predictions = test_predictions.copy()

    all_attention_weights = np.concatenate(all_attention_weights, axis=0).squeeze(-1)

    plt.figure(figsize=(10, 5))
    mean_attention = np.mean(all_attention_weights, axis=0)

    sns.barplot(x=np.arange(1, dp.LOOKBACK_WINDOW + 1), y=mean_attention, color='royalblue')
    plt.title('')
    plt.xlabel('День у вікні (1 - найстаріший, 21 - найближчий)')
    plt.ylabel('Середня вага')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    plt.figure(figsize=(14, 6))
    sns.heatmap(all_attention_weights[:].T, cmap='viridis', cbar_kws={'label': 'Вага'})
    plt.title('')
    plt.xlabel('Днів від початку тесту')
    plt.ylabel('День у вікні (лаг, 1 - найстаріший, 21 - найближчий)')
    plt.gca().invert_yaxis()
    plt.show()
    #----------------------------------------------------------
    # test_predictions = y_scaler.inverse_transform(test_predictions)
    # test_actuals = y_scaler.inverse_transform(test_actuals)

    direction_actuals = np.sign(test_actuals[1:] - test_actuals[:-1])
    m_direction_predictions = np.sign(test_predictions[1:] - test_predictions[:-1])

    m_mda = accuracy_score(direction_actuals, m_direction_predictions)

    test_naive_pred = test_actuals[:-1]
    test_naive_true = test_actuals[1:]

    n_mape = mean_absolute_percentage_error(test_naive_true, test_naive_pred)
    n_r2 = r2_score(test_naive_true, test_naive_pred)
    n_rmse = root_mean_squared_error(test_naive_true, test_naive_pred)

    m_mape = mean_absolute_percentage_error(test_actuals, test_predictions)
    m_r2 = r2_score(test_actuals, test_predictions)
    m_rmse = root_mean_squared_error(test_actuals, test_predictions)

    print('Extracting rolling GARCH...')
    garch_preds = data['garch'].tail(len(test_actuals)).to_numpy()

    g_mape = mean_absolute_percentage_error(test_actuals, garch_preds)
    g_r2 = r2_score(test_actuals, garch_preds)
    g_rmse = root_mean_squared_error(test_actuals, garch_preds)
    g_direction_predictions = np.sign(garch_preds[1:] - garch_preds[:-1])
    g_mda = accuracy_score(direction_actuals, g_direction_predictions)


    print('Calculating rolling HAR...')
    har_preds = np.array(har(train_data['garman-klass'], test_data['garman-klass']))

    h_mape = mean_absolute_percentage_error(test_actuals, har_preds)
    h_r2 = r2_score(test_actuals, har_preds)
    h_rmse = root_mean_squared_error(test_actuals, har_preds)
    h_direction_predictions = np.sign(har_preds[1:] - har_preds[:-1])
    h_mda = accuracy_score(direction_actuals, h_direction_predictions)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(test_actuals, color='red', label='Справжня', marker='.', ms=2)

    ax.plot(test_predictions, color='blue', label='LSTM-Attention', marker='.', ms=2, lw=2)

    ax.plot(garch_preds, color='orange', label='GARCH(1,1)', marker='.', ms=2, ls='--')

    ax.plot(har_preds, color='forestgreen', label='HAR-RV', marker='.', ms=2, ls='--', alpha=0.5)

    ax.set_xlabel('Днів від початку тесту')
    ax.set_ylabel('Волатильність Гарман-Класса')
    fig.text(0.1, 0.98,
             f'LSTM прогноз: MAPE: {m_mape:.4f}, R^2: {m_r2:.4f}, RMSE: {m_rmse:.4f}, MDA: {m_mda:.4f}')
    fig.text(0.1, 0.96,
             f'Наївний прогноз (t0 = t-1): MAPE: {n_mape:.4f}, R^2: {n_r2:.4f}, RMSE: {n_rmse:.4f}, MDA: -')

    fig.text(0.1, 0.94,f'GARCH прогноз: MAPE: {g_mape:.4f}, R^2: {g_r2:.4f}, RMSE: {g_rmse:.4f}, MDA: {g_mda:.4f}')

    fig.text(0.1, 0.92,f'HAR-RV прогноз: MAPE: {h_mape:.4f}, R^2: {h_r2:.4f}, RMSE: {h_rmse:.4f}, MDA: {h_mda:.4f}')

    fig.text(0.1, 0.02,
             'Джерело даних: Yahoo Finance | Курсова робота. Код доступний за адресою:'
             ' https://github.com/r0gerthaaat/VolatilityForecast')
    fig.text(0., 0.825,
             f'Params:\nbatch_size={BATCH_SIZE}\ninput_size={INPUT_SIZE}\nhidden_size={HIDDEN_SIZE}\n'
             f'num_layers={NUM_LAYERS}\ndropout={DROPOUT}\nlr={LEARNING_RATE}\nepochs={EPOCHS}\nwindow={dp.LOOKBACK_WINDOW}')
    plt.title('')
    plt.legend()
    plt.grid()
    plt.show()

    dm_stat, p_value = dm.dm_test(test_actuals, test_predictions, garch_preds, h=1, loss=lambda u, v: abs(u - v))
    print(f"DM Statistic LSTM/GARCH: {dm_stat:.4f}, P-value: {p_value:.4f}")

    dm_stat, p_value = dm.dm_test(test_actuals, test_predictions, har_preds, h=1, loss=lambda u, v: abs(u - v))
    print(f"DM Statistic LSTM/HAR: {dm_stat:.4f}, P-value: {p_value:.4f}")

    dm_stat, p_value = dm.dm_test(test_naive_true, test_predictions[1:], test_naive_pred, h=1, loss=lambda u, v: abs(u - v))
    print(f"DM Statistic LSTM/Naive: {dm_stat:.4f}, P-value: {p_value:.4f}")

    print(f'test has: {len(test_actuals)} actuals, {len(test_predictions)} preds')

    # DIFFERENCES UNROLLING (CHECK IF DIFF IS THE TARGET VALUE)
    if dp.USE_DIFFERENCES:
        fig2, ax2 = plt.subplots(figsize=(16, 9))

        true_baseline = test_data['garman-klass'].iloc[dp.LOOKBACK_WINDOW - 1]
        prev_gk = test_data['garman-klass'].iloc[
            dp.LOOKBACK_WINDOW - 1: dp.LOOKBACK_WINDOW - 1 + len(raw_diffs_predictions)].values

        test_actuals_unrolled = raw_diffs_actuals.cumsum() + true_baseline
        test_predictions_unrolled = raw_diffs_predictions + prev_gk

        m_mape = mean_absolute_percentage_error(test_actuals_unrolled, test_predictions_unrolled)
        m_r2 = r2_score(test_actuals_unrolled, test_predictions_unrolled)
        m_rmse = root_mean_squared_error(test_actuals_unrolled, test_predictions_unrolled)

        direction_predictions = np.sign(test_predictions_unrolled[1:] - test_predictions_unrolled[:-1])
        direction_actuals = np.sign(test_actuals_unrolled[1:] - test_actuals_unrolled[:-1])

        m_acc = accuracy_score(direction_actuals, direction_predictions)

        ax2.plot(test_actuals_unrolled, color='red', label='Справжня', marker='.', ms=2)

        ax2.plot(test_predictions_unrolled, color='blue', label='LSTM-Attention', marker='.', ms=2, lw=2)

        fig2.text(0.1, 0.98,
                 f'LSTM прогноз: MAPE: {m_mape:.4f}, R^2: {m_r2:.4f}, RMSE: {m_rmse:.4f}, вгадування напрямку: {m_acc:.4f}')
        ax2.set_xlabel('Днів від початку тесту')
        ax2.set_ylabel('Волатильність Гарман-Класса')
        plt.legend()
        plt.grid()
        plt.show()

    oc_ret = dp.get_CO_log_rets()
    test_oc_ret = oc_ret.tail(len(test_actuals)).to_numpy() * 100

    confidence_level = 0.95
    z_score = stats.norm.ppf(confidence_level)
    print(f'z-score: {z_score}')

    total_days = len(test_oc_ret)

    m_ivar_95 = z_score * test_predictions
    m_exceptions = test_oc_ret < -m_ivar_95
    m_num_exceptions = np.sum(m_exceptions)
    m_hit_rate = m_num_exceptions / total_days

    g_ivar_95 = z_score * garch_preds
    g_exceptions = test_oc_ret < -g_ivar_95
    g_num_exceptions = np.sum(g_exceptions)
    g_hit_rate = g_num_exceptions / total_days

    h_ivar_95 = z_score * har_preds
    h_exceptions = test_oc_ret < -h_ivar_95
    h_num_exceptions = np.sum(h_exceptions)
    h_hit_rate = h_num_exceptions / total_days

    n_ivar_95 = z_score * test_naive_pred
    n_exceptions = test_oc_ret[1:] < -n_ivar_95
    n_num_exceptions = np.sum(n_exceptions)
    n_hit_rate = n_num_exceptions / total_days

    print(f'Test days: {total_days}')
    print(f'LSTM IVaR hits: {m_num_exceptions}, Hit Rate: {m_hit_rate:.2%}')
    print(f'GARCH IVaR hits: {g_num_exceptions}, Hit Rate: {g_hit_rate:.2%}')
    print(f'HAR IVaR hits: {h_num_exceptions}, Hit Rate: {h_hit_rate:.2%}')
    print(f'Naive IVaR hits: {n_num_exceptions}, Hit Rate: {n_hit_rate:.2%}')


def har(train_vol, actual_test_vol):
    train = train_vol.to_numpy()
    actual_test = actual_test_vol.to_numpy()

    preds = []

    def fit_har(data):
        df = pd.DataFrame({'RV': data})
        df['RV_daily'] = df['RV'].shift(1)
        df['RV_weekly'] = df['RV'].shift(1).rolling(5).mean()
        df['RV_monthly'] = df['RV'].shift(1).rolling(22).mean()
        df = df.dropna()

        X = df[['RV_daily', 'RV_weekly', 'RV_monthly']]
        y = df['RV']

        return LinearRegression().fit(X, y)

    model = fit_har(train)

    for i in range(len(actual_test)):
        x_pred = pd.DataFrame({
            'RV_daily': [train[-1]],
            'RV_weekly': [train[-5:].mean()],
            'RV_monthly': [train[-22:].mean()]
        })

        pred_vol = model.predict(x_pred)[0]
        preds.append(pred_vol)

        train = np.append(train, actual_test[i])

        model = fit_har(train)

    return preds[dp.LOOKBACK_WINDOW - 1:]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random fixed: seed={seed}")


if __name__ == '__main__':
    main()