import pandas as pd
import numpy as np
from arch import arch_model

def rolling_garch(returns: pd.Series, window: int):
    forecasts = [np.nan] * window

    for i in range(window, len(returns)):
        returns_window = returns.iloc[i - window + 1 : i + 1]

        model = arch_model(y=returns_window, p=1, q=1, vol='GARCH', dist='t').fit(disp='off')
        res = model.forecast(horizon=1)
        forecasts.append(np.sqrt( res.variance.values[-1, 0] ))

    return forecasts
