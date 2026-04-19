# Phase II Submission: Bitcoin Returns Forecasting using Temporal Fusion Transformers

## Overview
This repository contains the Phase II final submission notebook for the Bitcoin (BTC) forecasting project. The objective is to predict daily log returns of Bitcoin using a deep learning time-series architecture. We utilize a **Temporal Fusion Transformer (TFT)** built with `pytorch-forecasting` and `lightning.pytorch` to model complex temporal dynamics, evaluate predictive accuracy, and backtest a simulated algorithmic trading strategy.

## Dependencies
To execute this notebook, the following libraries are required:
* `pytorch-forecasting`
* `pytorch-lightning`
* `pytorch-optimizer`
* `pandas` & `numpy`
* `scikit-learn`
* `matplotlib` & `seaborn`

## Data Processing & Feature Engineering
The pipeline begins with a raw timestamped Bitcoin dataset (`btc_data.csv`), filtered for records from 2017 onwards. 
The data is resampled to a daily frequency (OHLCV) to mitigate intraday noise. The following features are engineered to provide the model with market context:
* **Log Returns:** The primary target variable, ensuring stationarity.
* **7-Day Volatility:** Rolling standard deviation of the log returns to capture local market turbulence.
* **14-Day RSI (Relative Strength Index):** A momentum oscillator to measure the speed and magnitude of recent price changes.
* **Day of Week:** Extracted as a static categorical variable to account for weekend/weekday trading discrepancies.

## Dataset Preparation
The data is fed into a `TimeSeriesDataSet` utilizing:
* **Max Encoder Length:** 30 days (historical lookback window).
* **Max Prediction Length:** 7 days (forecast horizon).
* **Time-varying unknown variables:** Log returns, close price, volume, volatility, and RSI.
* **Out-of-Sample Validation:** The validation dataset strictly relies on chronological unseen data, enforcing a 1-step-ahead rolling prediction to prevent data leakage.

## Model Architecture
The baseline model is a **Temporal Fusion Transformer (TFT)** configured with:
* Hidden Size: 64
* Attention Heads: 4
* Dropout: 0.1
* Optimizer: `Ranger` (Learning rate: 1e-3)
* Loss Function: Root Mean Squared Error (RMSE)

*Note: The notebook is configured to bypass training and load pre-optimized weights from a mounted Google Drive checkpoint (`tft-baseline-epoch=00-val_loss=0.0287.ckpt`) for rapid evaluation.*

## Evaluation & Portfolio Risk Metrics
The model's predictions are evaluated strictly on the **1-step-ahead** forecast (the immediate next day) to simulate a realistic live trading environment. 

### Regression Metrics:
* **MAE (Mean Absolute Error):** Measures the average magnitude of forecasting errors.
* **RMSE (Root Mean Squared Error):** Penalizes larger forecasting errors.
* **R² Score:** Assesses the proportion of variance in the target variable explained by the model.
* **Directional Accuracy:** The percentage of days the model correctly guesses the sign (+/-) of the market move.

### Strategy Backtesting Metrics:
Trading signals are generated based on the sign of the model's prediction (+1 for Long, -1 for Short). The backtest yields the following institutional risk metrics:
* **Annualized Sharpe Ratio:** Risk-adjusted return calculation normalized for the 24/7 crypto market (365 days).
* **Maximum Drawdown:** The largest peak-to-trough drop in the simulated equity curve.
* **Cumulative Return:** The total compounded return of the Long/Short strategy over the out-of-sample period.

## Visualizations
The notebook generates quantitative visualizations using `matplotlib` and `seaborn`:
1. **Time-Series Plot:** A chronological overlay of Predicted vs. Actual Log Returns (including a directional pivot zero-line).
2. **Scatter Plot:** Predicted vs. Actual Returns with a line of best fit to visually assess correlation and outlier impact.
