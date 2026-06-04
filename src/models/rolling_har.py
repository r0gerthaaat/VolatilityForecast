import pandas as pd
from sklearn.linear_model import LinearRegression
import src.data.data_preprocessor as dp

def rolling_har(train_vol: pd.Series, actual_vol: pd.Series, window: int):
    vol = pd.concat([train_vol, actual_vol])

    data = pd.DataFrame({
        'RV': vol,
        'RV_daily': vol.shift(1),
        'RV_weekly': vol.shift(1).rolling(5).mean(),
        'RV_monthly': vol.shift(1).rolling(22).mean()
    })

    data.dropna(inplace=True)

    X = data.drop(columns=['RV'])
    y = data['RV']

    predictions = []

    test_size = len(actual_vol)
    total_size = len(data)
    train_size = total_size - test_size

    for i in range(len(actual_vol)):
        current_step = train_size + i

        X_train = X.iloc[current_step - window : current_step]
        y_train = y.iloc[current_step - window : current_step]

        lr = LinearRegression().fit(X_train, y_train)

        pred = lr.predict(X.iloc[[current_step]])
        predictions.append(pred[0])

    return predictions[dp.LOOKBACK_WINDOW - 1:]