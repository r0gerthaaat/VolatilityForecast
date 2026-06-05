# S&P 500 Intraday Volatility Forecasting

> PyTorch · LSTM · Attention mechanism · GARCH / HAR-RV / Naive benchmarks · Academic project

Deep learning approach to forecasting S&P 500 intraday volatility using a two-layer LSTM network with an Attention mechanism, benchmarked against classical econometric baselines. Developed as an academic research supervised by Prof. Andrii Stavytskyi, Taras Shevchenko National University of Kyiv. 

<img width="700" height="417" alt="GK-vol-forecast" src="https://github.com/user-attachments/assets/7d141b8e-5f55-4e36-9807-e19cb09e77c2" />

---

## Stack

| Layer | Tool |
| --- | --- |
| Language | Python |
| Deep learning | `PyTorch` |
| Data & preprocessing | `yfinance`, `pandas`, `NumPy`, `scikit-learn` |
| Econometrics | `arch`, `statsmodels` for ADF test, `dieboldmariano` for DM test |
| Dependency management | `uv` |

---

## Approach

### Target variable

Instead of using the standard deviation of log returns, the model targets **Garman-Klass Volatility** - an estimator that incorporates the full OHLC range and therefore captures intraday dynamics more accurately when working with daily bars. The VIX index is included as an exogenous feature to supply the model with forward-looking market expectations.

The GK estimator is defined as:

$$\sigma_t^{GK} = \sqrt{\frac{1}{2}\ln\left(\frac{h_t}{l_t}\right)^{2} - (2\ln 2 - 1)\ln\left(\frac{c_t}{o_t}\right)^{2}}$$

### Model architecture

The architecture consists of a two-layer LSTM (hidden size 256) followed by an Attention layer and a linear output head.

<img width="511" height="287" alt="lstm-att architecture" src="https://github.com/user-attachments/assets/1e4d5bda-a0c9-4567-bd73-a204d14b4265" />


The LSTM component handles long-term regime memory and short-term shock persistence. The core motivation for the Attention layer is the information bottleneck problem in standard RNNs: compressing an entire sequence into a single fixed-size vector causes older, potentially relevant states to be forgotten. The attention mechanism instead computes a weighted sum over all hidden states within the 21-day lookback window, allowing the model to dynamically focus on specific historical days rather than treating all of them uniformly.

<img width="576" height="297" alt="attn-weights" src="https://github.com/user-attachments/assets/2b327d70-c772-4b2c-a130-46416391beb9" />

Training uses **Huber Loss** to handle the heavy tails common in financial volatility series - it behaves as MSE for small errors and as MAE for large ones, preventing extreme observations from dominating the gradient signal. 

QLIKE was tested as a loss function, but it seems to make the gradient too steep so that model often cathes local minima.

---

## Results

Models were trained and evaluated on S&P 500 daily data from 2015 (2016 considering the GARCH training period of 250 trading days) to 2025 using a simulated ex-ante rolling forecast scheme, so no future information leaks into the predictions.

> Note: GARCH is inefficient at forecasting the original GK series due to a structural mismatch: GARCH is fitted on close-to-close returns, which incorporate both intraday movement and overnight gaps, while the GK estimator captures intraday dynamics only (open-to-close range). As a result, GARCH systematically overestimates the GK target.

### Regression metrics (original series)

| Model | RMSE | MAPE | R² | MDA |
| --- | --- | --- | --- | --- |
| **LSTM-Attention** | **0.3173** | **0.3460** | **0.7230** | 0.4675 |
| GARCH(1,1) | 0.6901 | 0.8362 | -0.3097 | **0.5152** |
| HAR-RV | 0.4993 | 0.4249 | 0.3143 | 0.4675 |
| Naive (t₀ = t−1) | 0.4350 | 0.3880 | 0.4818 | - |

The LSTM-Attention's advantage over both classical approaches is statistically confirmed by the Diebold-Mariano test (p-value < 0.01).

Forecasting first-differences of the series boosts directional accuracy (MDA) to **78.54%**, but after unrolling predictions back to the original scale the regression metrics deteriorate - stripping the model of its long-term context is not worth the trade-off.

### Regression metrics (differences)

The other approach that targets the GK differences (delta) and then unrolls the series was tested:

$$\hat{\sigma}_t^{GK} = \widehat{\Delta\sigma}_t^{GK} + \sigma_{t-1}^{GK}$$

| Approach | RMSE | MAPE | R² | MDA |
| --- | --- | --- | --- | --- |
| **LSTM-Attention on original series** | **0.3173** | **0.3460** | **0.7230** | **0.4675** |
| LSTM-Attention on differences series | 0.3203 | 0.3569 | 0.7020 | 0.4494 |

<img width="712" height="425" alt="GK-diff-vol-forecast" src="https://github.com/user-attachments/assets/2db00393-80e1-4f29-8518-c1ea46729862" />

Though on the differences series itself it shows MDA of 78,5%, no practical improvement showed after series unrolling.

### Risk assessment: Intraday Value at Risk (IVaR)

An ex-post IVaR analysis at the 95% confidence level was run to evaluate practical utility. The theoretical target hit rate is **5.00%**.

| Model | Hit Rate | Assessment |
| --- | --- | --- |
| LSTM-Attention | 5.60% | Close to target, adequate risk coverage |
| GARCH(1,1) | 2.59% | Excessively conservative, systematically overestimates risk |
| HAR-RV | 6.47% | Underestimates risk |
| **Naive** | **4.74%** | Closest to target, but forecasting metrics are weaker from NN |

---

## Repo structure

```
data/           Datasets with preloaded features and generated GK volatility targets
plots/          Loss curves, forecast vs. actual series, attention weight heatmaps
src/
  data/         Data loading and preprocessing pipelines
  models/       VolatilityLSTM architecture, rolling GARCH and HAR implementations
  evaluation/   Custom loss functions (QLIKE) and Dickey-Fuller test
main.py         Full pipeline: data prep, training loop (Adam), validation, IVaR
```

---

## How to run

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/r0gerthaaat/VolatilityForecast.git
cd VolatilityForecast
uv sync
python main.py
```

`uv` resolves and installs the full environment from `pyproject.toml`. No manual pip installs required.

Check the `USE_DIFFERENCES` boolean in ```src/data/datapreprocessor.py``` to run for the differences forecast.

---

## Known limitations

- **Directional accuracy** - despite substantially outperforming GARCH and HAR-RV on regression metrics, the LSTM-Attention's MDA on the original series (46.75%) does not beat GARCH (51.52%), suggesting the model optimizes point estimates rather than sign prediction
- **Regime generalization** - the rolling ex-ante scheme does not perfectly simulate live deployment; extreme tail events that fall far outside the training distribution (e.g., sudden liquidity crises) may degrade performance in ways the backtest doesn't capture, though the Trump`s tariffs volatility spike on the ~50 day of the test was correctly predicted
- **Static lookback window** - the 21-day window is fixed; an adaptive or learned window length could improve performance on regimes with unusually long memory
- **Single asset** - the model is trained and tested on S&P 500 only; generalizability to other assets or markets is untested

---

*Denys Kovika · 2nd year, Economic Cybernetics, Taras Shevchenko National University of Kyiv*
