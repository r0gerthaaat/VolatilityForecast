import pandas as pd
from arch import arch_model

import data_preprocessor as dp

def rolling_garch(returns: pd.Series, window: int) -> pd.Series:
    model = arch_model(y=returns, p=1, q=1, vol='GARCH', dist='t').fit(disp='off')
