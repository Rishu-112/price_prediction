# Interview Guide – Stock Price Prediction (LSTM + Streamlit)

## 30-second pitch

I built a Streamlit web app that pulls live stock data from Yahoo Finance (`yfinance`), shows candlestick/volume/MA50/MA200 charts, compares an LSTM model's predictions with actual prices, reports RMSE and a MAPE-based accuracy, and forecasts the next 100 business days.

## Architecture

```
yfinance (2012 → today)
   │
   ▼
Close prices ──► 80/20 chronological split ──► MinMaxScaler (0–1)
   │
   ▼
Sliding windows: 100 past days ──► next day's close
   │
   ▼
LSTM(50) → LSTM(50) → LSTM(50) → Dense(1)   (Adam, MSE, 50 epochs, batch 64)
   │
   ▼
Inverse-scale ──► charts + RMSE / MAPE ──► recursive 100-day forecast
```

| File | Purpose |
|---|---|
| `Stock_Market_Prediction_Model_Creation_executed.ipynb` | Data exploration, training, saving the model |
| `Stock Prediction.keras` | Trained model loaded by the app |
| `app_final.py` | Streamlit app (data, charts, evaluation, forecast) |
| `predicted_stock_prices.csv` | Exported predictions |

## Key design decisions (and how to justify them)

- **Why LSTM?** Prices are sequential; LSTMs keep memory across a window via gates, and avoid the vanishing-gradient problem of plain RNNs.
- **Why a 100-day window?** It gives enough context for trends (~5 months of trading) without making training too slow. It's a hyperparameter I'd tune.
- **Why MinMaxScaler?** LSTMs train better on small, bounded inputs (sigmoid/tanh gates).
- **Why a chronological split?** Random splits leak future information into training; time series need train-before-test.
- **Why prepend the last 100 training days to the test set?** The first test sample needs 100 days of history.
- **Why MSE loss + Adam?** Standard for regression; Adam adapts learning rates and converges quickly.

## Likely questions

**1. Walk me through the pipeline.**
Download data → compute MA50/MA200 → split 80/20 by time → scale → build 100-day windows → train a 3-layer LSTM → predict → inverse-scale → evaluate and visualise → forecast.

**2. How does the 100-day future forecast work?**
Recursively: predict one day, append the prediction to the input window, drop the oldest day, repeat 100 times. Errors compound, so long-horizon forecasts are unreliable and should be treated as an illustration of the model's learned trend, not as a trading signal.

**3. How do you evaluate it?**
RMSE (in dollars, penalises large errors) and MAPE (percentage error, scale-independent). The app shows accuracy as `100 − MAPE`.

**4. Can this actually predict the market?**
Not reliably. One-step-ahead prediction looks accurate largely because tomorrow's price is close to today's; a naive "tomorrow = today" baseline is hard to beat. Markets are driven by news and other factors that a price-only model can't see. I'd say this honestly in an interview — it shows maturity.

**5. How did you prevent overfitting?**
Honest answer: the current model has no dropout, early stopping, or validation split. Improvements I'd make: add `Dropout`, `EarlyStopping` on a validation set, and walk-forward validation.

**6. What would you improve?** (see below)

## Known limitations (be upfront about these)

1. **Scaler is re-fit on test data** (`fit_transform` on the test and again on the train set in the app). Correct practice is to fit the scaler on training data only and `transform` the rest; otherwise information leaks and the scaling differs from what the model was trained with.
2. **Accuracy metric is simplistic.** `sklearn`'s MAPE returns a fraction, so `100 − MAPE` is not a true percentage accuracy. Reporting MAPE as a percentage (×100) and comparing to a naive baseline would be more honest.
3. **Univariate input.** Only `Close` is used – no volume, indicators, sentiment, or macro data.
4. **No baseline comparison** (naive, ARIMA, moving average).
5. **Recursive forecasting drifts** and error accumulates over 100 steps.
6. **Model is trained once** (in the notebook, on one ticker) and reused for all tickers in the dropdown, with the scaler fitted per ticker at runtime.
7. **App re-downloads data and re-runs predictions on every interaction** – no caching (`st.cache_data` / `st.cache_resource`).

## Improvements roadmap

- Fit scaler on train only; save it with the model.
- Add Dropout, EarlyStopping, validation split, walk-forward validation.
- Compare against naive / ARIMA / GRU / Transformer baselines.
- Add features: volume, RSI, MACD, MA, sentiment.
- Predict returns instead of raw prices (more stationary).
- Cache the model and downloaded data in Streamlit.
- Add prediction intervals (e.g. Monte Carlo dropout) to show uncertainty.
- Add tests and CI; containerise with Docker and deploy (Streamlit Cloud).

## Quick concept refresher

- **LSTM gates:** forget (what to drop), input (what to add), output (what to expose); the cell state carries long-term memory.
- **return_sequences=True:** returns the full sequence so the next LSTM layer can stack on it; the last LSTM returns only the final state.
- **Input shape `(samples, 100, 1)`:** samples × timesteps × features.
- **RMSE vs MAPE:** RMSE is in price units and sensitive to outliers; MAPE is relative and unstable when prices are near zero.
- **MA50 / MA200:** short- and long-term trend indicators; a MA50 crossing above MA200 is the "golden cross".
- **Why not shuffle?** Temporal order matters; shuffling leaks the future.

## Demo script (2 minutes)

1. Open the app, pick a ticker (e.g. `AAPL`).
2. Show the candlestick chart with volume and MA50/MA200.
3. Show actual vs predicted – explain train and test regions.
4. Point to RMSE / accuracy and say what they mean (and don't mean).
5. Show the 100-day forecast and explain the recursive approach and its caveat.
6. Close with limitations and what you'd improve next.
