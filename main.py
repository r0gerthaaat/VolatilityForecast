import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import data_preprocessor as dp
from dataset import VolatilityDataset
from model import VolatilityLSTM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from arch import arch_model

import random
import os

BATCH_SIZE = 256

INPUT_SIZE = 10
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

LEARNING_RATE = 0.001
EPOCHS = 89

USE_GARCH = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def main():
    seed_everything()

    # DATA PREPARING BLOCK
    dp.LOOKBACK_WINDOW = 5
    data = dp.preprocess_data()
    print(f'{len(data)} rows')

    train_data, test_data = train_test_split(data, test_size=0.1, shuffle=False)
    train_x = train_data.drop(columns=['target'])
    train_y = train_data[['target']]

    test_x = test_data.drop(columns=['target'])
    test_y = test_data[['target']]

    x_scaler: StandardScaler = StandardScaler().fit(train_x)
    y_scaler: StandardScaler = StandardScaler().fit(train_y)

    train_x_scaled: np.ndarray = x_scaler.transform(train_x)
    train_y_scaled: np.ndarray = y_scaler.transform(train_y)

    test_x_scaled: np.ndarray = x_scaler.transform(test_x)
    test_y_scaled: np.ndarray = y_scaler.transform(test_y)

    train_ds = VolatilityDataset(train_x_scaled, train_y_scaled, dp.LOOKBACK_WINDOW)
    test_ds = VolatilityDataset(test_x_scaled, test_y_scaled, dp.LOOKBACK_WINDOW)

    train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # shuffle for pattern recognizing instead of chronologic remembering
    test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # MODEL BLOCK
    model = VolatilityLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    criterion = nn.HuberLoss()
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

    sns.lineplot(train_losses, color='red')
    sns.lineplot(test_losses, color='blue')
    plt.title('Losses')
    plt.show()

    test_predictions = []
    test_actuals = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            out = model(x_batch)

            out = out.cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy()

            test_predictions.append(out)
            test_actuals.append(y_batch)


    test_predictions = np.vstack(test_predictions)
    test_actuals = np.vstack(test_actuals)

    test_predictions = y_scaler.inverse_transform(test_predictions)
    test_actuals = y_scaler.inverse_transform(test_actuals)

    direction_predictions = np.sign(test_predictions[1:] - test_predictions[:-1])
    direction_actuals = np.sign(test_actuals[1:] - test_actuals[:-1])

    m_acc = accuracy_score(direction_actuals, direction_predictions)

    test_naive_pred = test_actuals[:-1]
    test_naive_true = test_actuals[1:]

    n_mape = mean_absolute_percentage_error(test_naive_true, test_naive_pred)
    n_r2 = r2_score(test_naive_true, test_naive_pred)
    n_rmse = root_mean_squared_error(test_naive_true, test_naive_pred)

    m_mape = mean_absolute_percentage_error(test_actuals, test_predictions)
    m_r2 = r2_score(test_actuals, test_predictions)
    m_rmse = root_mean_squared_error(test_actuals, test_predictions)


    print('Calculating rolling GARCH...')
    garch_preds = rolling_garch(train_data['log_return'], test_data['log_return'])

    g_mape = mean_absolute_percentage_error(test_actuals, garch_preds)
    g_r2 = r2_score(test_actuals, garch_preds)
    g_rmse = root_mean_squared_error(test_actuals, garch_preds)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(test_actuals, color='red', label='Справжня', marker='.', ms=2)
    ax.plot(test_predictions, color='blue', label='LSTM без уваги', marker='.', ms=2)

    ax.plot(garch_preds, color='orange', label='GARCH(1,1)', marker='.', ms=2)

    ax.set_xlabel('Днів від початку тесту')
    ax.set_ylabel('Волатильність')
    fig.text(0.1, 0.98,
             f'LSTM прогноз: MAPE: {m_mape:.4f}, R^2: {m_r2:.4f}, RMSE: {m_rmse:.4f}, вгадування напрямку: {m_acc:.4f}')
    fig.text(0.1, 0.96,
             f'Наївний прогноз (t0 = t-1): MAPE: {n_mape:.4f}, R^2: {n_r2:.4f}, RMSE: {n_rmse:.4f}')

    fig.text(0.1, 0.94,f'GARCH прогноз: MAPE: {g_mape:.4f}, R^2: {g_r2:.4f}, RMSE: {g_rmse:.4f}')

    fig.text(0.1, 0.02,
             'Джерело даних: Yahoo Finance | Курсова робота. Код доступний за адресою:'
             ' https://github.com/r0gerthaaat/VolatilityForecast')
    fig.text(0., 0.825,
             f'Params:\nbatch_size={BATCH_SIZE}\ninput_size={INPUT_SIZE}\nhidden_size={HIDDEN_SIZE}\n'
             f'num_layers={NUM_LAYERS}\ndropout={DROPOUT}\nlr={LEARNING_RATE}\nepochs={EPOCHS}')
    plt.title('Прогнозування волатильності')
    plt.legend()
    plt.show()


def rolling_garch(train_returns, actual_test_returns):
    train = train_returns.to_numpy() * 100
    actual_test = actual_test_returns.to_numpy() * 100

    preds = []
    model = arch_model(train, p=2, q=2, vol='Garch', dist='t').fit(disp='off')

    for i in range(len(actual_test)):
        train = np.append(train, actual_test[i])

        fc = model.forecast(horizon=1)

        pred_vol = np.sqrt(fc.variance.values[-1, 0])
        preds.append(pred_vol)

        model = arch_model(train, p=2, q=2, vol='Garch', dist='t').fit(disp='off')
    return preds[dp.LOOKBACK_WINDOW-1:]


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