#!/usr/bin/env python3
# 1D_Momentum_RSI_Trend_Filter
# Hypothesis: Use daily RSI for momentum with 200-day EMA trend filter and volume confirmation.
# Enter long when RSI crosses above 50, price above EMA200, and volume above average.
# Enter short when RSI crosses below 50, price below EMA200, and volume above average.
# Exit when RSI crosses back to 50 or price crosses EMA200 in opposite direction.
# Designed to capture momentum shifts with trend alignment, avoiding counter-trend trades.
# Target: 15-25 trades/year on 1d to minimize fee decay while capturing sustained moves.

name = "1D_Momentum_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate RSI(14) on daily closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))

    # Calculate EMA200 for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values

    # Weekly EMA50 for higher timeframe trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema200[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI crosses above 50 + price above EMA200 + weekly uptrend + volume spike
            if (rsi[i] > 50 and rsi[i-1] <= 50 and
                close[i] > ema200[i] and
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below 50 + price below EMA200 + weekly downtrend + volume spike
            elif (rsi[i] < 50 and rsi[i-1] >= 50 and
                  close[i] < ema200[i] and
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below 50 or price crosses below EMA200
            if (rsi[i] < 50 and rsi[i-1] >= 50) or (close[i] < ema200[i] and close[i-1] >= ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above 50 or price crosses above EMA200
            if (rsi[i] > 50 and rsi[i-1] <= 50) or (close[i] > ema200[i] and close[i-1] <= ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals