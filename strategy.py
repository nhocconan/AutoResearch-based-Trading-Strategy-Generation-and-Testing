#!/usr/bin/env python3
"""
1d_TRIX_ZeroCross_WeekTrend_Volume
Hypothesis: On 1d timeframe, TRIX(9) crossing zero with 1-week EMA trend and volume > 1.5x 20-day average provides reliable momentum entries. 
TRIX filters noise by focusing on momentum changes; weekly trend ensures alignment with higher timeframe direction. 
Volume confirmation avoids low-liquidity breakouts. 
Exit on reverse TRIX cross or trend violation. 
Designed for low turnover (target: 10-25 trades/year) to minimize fee drag in 2025+ bearish/range markets.
Works in bull via momentum acceleration and bear via mean-reversion at trend extremes.
"""

name = "1d_TRIX_ZeroCross_WeekTrend_Volume"
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

    # Get 1w data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate TRIX(9) on daily close: triple EMA of ROC
    # ROC = (close/t - close/t-1) / close/t-1
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1]
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = ema3 * 100  # scale for readability
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = np.nan

    # Calculate 1-week EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current day
        trix_now = trix[i]
        trix_prev_val = trix_prev[i]
        ema34 = ema34_1w_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(trix_now) or np.isnan(trix_prev_val) or 
            np.isnan(ema34) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + price above weekly EMA34 + volume surge
            if (trix_prev_val <= 0 and trix_now > 0 and 
                close[i] > ema34 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + price below weekly EMA34 + volume surge
            elif (trix_prev_val >= 0 and trix_now < 0 and 
                  close[i] < ema34 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or price below weekly EMA34
            if (trix_prev_val >= 0 and trix_now < 0) or close[i] < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or price above weekly EMA34
            if (trix_prev_val <= 0 and trix_now > 0) or close[i] > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals