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

BATCH_SIZE = 32

INPUT_SIZE = 8
HIDDEN_SIZE = 16
NUM_LAYERS = 1
DROPOUT = 0.5

LEARNING_RATE = 0.0005
EPOCHS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def main():
    # DATA PREPARING BLOCK
    dp.LOOKBACK_WINDOW = 60
    data = dp.preprocess_data()
    print(len(data))

    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    train_losses = []
    test_losses = []

    print('Start training')

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

    acc = accuracy_score(direction_actuals, direction_predictions)

    test_naive_pred = test_actuals[:-1]
    test_naive_true = test_actuals[1:]

    n_mape = mean_absolute_percentage_error(test_naive_true, test_naive_pred)
    n_r2 = r2_score(test_naive_true, test_naive_pred)
    n_rmse = root_mean_squared_error(test_naive_true, test_naive_pred)

    m_mape = mean_absolute_percentage_error(test_actuals, test_predictions)
    m_r2 = r2_score(test_actuals, test_predictions)
    m_rmse = root_mean_squared_error(test_actuals, test_predictions)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(test_actuals, color='red', label='Справжнє')
    ax.plot(test_predictions, color='blue', label='Прогноз')
    ax.set_xlabel('Днів від початку тесту')
    ax.set_ylabel('Волатильність = STDEV(від t+1 до t+5)')
    fig.text(0.08, 0.98,
             f'Модельний прогноз: MAPE: {m_mape:.4f}, R^2: {m_r2:.4f}, RMSE: {m_rmse:.4f}, вгадування напрямку: {acc:.4f}')
    fig.text(0.08, 0.96,
             f'Наївний прогноз (t0 = t-1): MAPE: {n_mape:.4f}, R^2: {n_r2:.4f}, RMSE: {n_rmse:.4f}')
    fig.text(0.08, 0.02,
             'Джерело: Yahoo Finance | Автор: Денис Ковіка, студент кафедри Економічної кібернетики')
    plt.title('Прогнозування LSTM моделлю з 1 шаром')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()