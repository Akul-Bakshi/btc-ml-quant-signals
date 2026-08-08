# Architecture Overview

## Project Structure
```text
btc-ml-bot/
├── ARCHITECTURE.md          # Project architecture and documentation
├── config.yaml              # Global configuration (timeframes, risk, strategies)
├── requirements.txt         # Python dependencies
├── data/                    # Data ingestion and processing
│   ├── fetcher.py           # Downloads OHLCV data from Binance
│   ├── cleaner.py           # Cleans raw data and interpolates missing values
│   ├── features.py          # Computes technical indicators and target labels
│   ├── raw/                 # Stored raw and cleaned CSVs from fetcher/cleaner
│   └── processed/           # Stored ML-ready CSVs from features.py
├── models/                  # Machine learning models and training logic
│   ├── train.py             # Training script for standard models (e.g., 1h swing)
│   ├── train_scalper.py     # Specialized script for training 15m scalper models
│   ├── predict.py           # Inference script for real-time predictions
│   ├── evaluate.py          # ML evaluation and cross-validation utilities
│   └── saved/               # Stored artifacts (.pkl models, feature json, backtest logs)
├── strategy/                # Trading logic and simulation
│   ├── signals.py           # Regime detection (TREND, RANGE, CHOP) and signal generation
│   ├── risk.py              # Position sizing and risk management logic
│   ├── backtest.py          # Historical simulation engine
│   └── genetic.py           # Genetic algorithm for parameter optimization
├── dashboard/               # Streamlit user interfaces
│   ├── app.py               # Main real-time market data and indicator visualization
│   └── backtest_viz.py      # Backtest trades, equity curve, and performance visualization
└── bot/                     # Live trading execution and tracking
    ├── live_tracker.py      # Live forward-testing script (tracks signals in real-time)
    ├── live_signals.csv     # (Generated) Log of all live signals and their outcomes
    └── live_stats.json      # (Generated) Running tally of live forward-test performance
```

## Detailed File Responsibilities

### `/data/` (Data Pipeline)
- **`fetcher.py`**: Connects to the Binance API to download historical OHLCV data for configured symbols (BTC/USDT, ETH/USDT) and timeframes. Saves data to `/data/raw/`.
- **`cleaner.py`**: Loads raw data, removes duplicates, handles missing rows, and ensures chronological integrity. Outputs `_clean.csv` to `/data/raw/`.
- **`features.py`**: The core feature engineering script. Calculates momentum (RSI, Stoch, ROC), trend (EMA), volatility (Bollinger Bands, ATR), volume, and custom statistical features (Z-scores). It also builds the ML target labels (e.g., `target_direction` based on forward returns). It merges higher timeframe context (1h, 4h, 1d) onto the base timeframe and saves `_features.csv` to `/data/processed/`.

### `/models/` (Machine Learning)
- **`train_scalper.py`**: Orchestrates the training of an ensemble of models (Logistic Regression, Random Forest, XGBoost, LightGBM) specifically tuned for the 15m timeframe. It implements walk-forward cross-validation, selects the top 25 features, and serializes the best ensemble and feature list into `/models/saved/`.
- **`predict.py`**: Loads the serialized model and selected features to predict the probability of a successful trade on live/recent data.

### `/strategy/` (Trading Logic)
- **`signals.py`**: The brain of the trading strategy.
  - `detect_regime()` evaluates 4h ADX, EMA separation, Bollinger Bandwidth, and ROC to classify the market as `TREND`, `RANGE`, or `CHOP`.
  - Depending on the regime, it routes to `get_trend_entry()` or `get_range_entry()`.
  - It queries the ML model (from `predict_proba()`) and only returns a valid signal if the ML confidence exceeds the threshold (e.g., `0.62`).
- **`backtest.py`**: The simulation engine. It iterates over the 15m `_features.csv` data row by row, keeping track of the 1h and 4h context windows. It calls `signals.py` for entries, manages open positions (checking for stop loss, take profit, or time-based exits), updates the equity curve, and exports the final metrics, trade logs, and price series JSONs to `/models/saved/`.
- **`risk.py`**: Handles position sizing logic, ensuring spot trades do not exceed 95% of available balance and that the nominal risk per trade adheres to the target percentage (e.g., 2%).

### `/dashboard/` (Visualization)
- **`app.py`**: A robust Streamlit dashboard for exploring the processed data. Contains a "LIVE TRACKING" panel to monitor out-of-sample forward-tests in real-time alongside historical backtests. The main chart actively fetches live market data directly from the exchange (via CCXT public endpoints) and computes indicators on the fly.
- **`backtest_viz.py`**: A specialized Streamlit app for backtest analysis.

### `/bot/` (Execution & Tracking)
- **`live_tracker.py`**: A standalone script that queries the Binance API (via `ccxt` public endpoints) for the latest market data, passes it through the exact feature pipeline, queries the ML model, and logs trading signals to `live_signals.csv`. It actively manages the status of open signals by checking whether targets or stops were hit and tallies the performance in `live_stats.json`.

## Data & Execution Flow
1. **Preparation**: `fetcher.py` -> `cleaner.py` -> `features.py`. Raw data is downloaded, cleaned, and heavily engineered with cross-timeframe indicators.
2. **Training**: `train_scalper.py` reads the 15m `_features.csv`, applies walk-forward validation, and saves `scalper_ensemble.pkl` and `scalper_selected_features.json`.
3. **Simulation**: `backtest.py` runs over the 15m dataset.
   - For every candle, it aligns the latest 1h and 4h data.
   - It calls `signals.py` -> `detect_regime(4h)`.
   - If not `CHOP`, it checks entry criteria on 15m/1h.
   - If entry criteria pass, it checks ML confidence.
   - If ML confidence >= `0.62`, a position is opened.
   - Open positions are evaluated every 15m candle against dynamic stop-loss, take-profit, or a maximum time-in-trade limit.
   - Completed trades and equity tracking are saved to disk.
4. **Review**: `dashboard/backtest_viz.py` renders the results.

## Key Config Values (`config.yaml`)
- **Timeframes**: Entry = 15m, Confirmation = 1h, Direction = 1d/4h.
- **Risk per Trade**: 2.0% of balance.
- **Spot Constraints**: Maximum position size capped at 95% of account balance (no leverage).
- **ML Confidence**: Desired minimum confidence threshold is defined as 0.65 (Note: actual execution relies on overrides in `signals.py`).

## Current Known Issues
1. **Missing Data in ML Pipeline**: Approximately 77% of 15m candles are dropped in `features.csv`. This is caused by a strict `abs(target_return) >= 0.003` filter during feature generation (`features.py`), which aggressively discards any candle that didn't experience a 0.3% price move in the next 2 periods.
2. **Hardcoded Strategy Thresholds**: Several critical configuration values (e.g., ML confidence threshold of 0.62, TREND/RANGE specific stop-loss and take-profit percentages) are hardcoded directly inside `strategy/signals.py` rather than being dynamically injected from `config.yaml`. 
3. **Model Overfitting**: The LightGBM and XGBoost models currently show noticeable overfitting on the 15m timeframe (e.g., LightGBM has ~10% gap between train and test accuracy), though Random Forest remains stable.
