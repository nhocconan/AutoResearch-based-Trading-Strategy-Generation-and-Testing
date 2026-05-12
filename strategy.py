#!/usr/bin/env python3
"""
12h_1W_TRIX_Trend_With_Volume_Confirmation
Hypothesis: On 12-hour timeframe, TRIX (15-period) crossing zero with
weekly EMA34 trend filter and volume > 1.5x 50-period average generates
high-probability trend-following signals. TRIX momentum combined with
weekly trend alignment and volume confirmation filters whipsaws.
Designed to work in both bull and bear markets via weekly trend filter
and volume confirmation requirement. Targets 12-37 trades/year.
"""

name = "12h_1W_TRIX_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate TRIX (15-period triple EMA)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values

    # Volume confirmation: 1.5x 50-period average
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 12h bar
        ema34 = ema34_1w_aligned[i]
        vol_avg_val = vol_avg_50[i]
        trix_val = trix[i]
        trix_prev = trix[i-1]

        # Skip if any required data is NaN
        if (np.isnan(ema34) or np.isnan(vol_avg_val) or 
            np.isnan(trix_val) or np.isnan(trix_prev)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + weekly uptrend + volume surge
            if (trix_prev <= 0 and trix_val > 0 and 
                close[i] > ema34 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + weekly downtrend + volume surge
            elif (trix_prev >= 0 and trix_val < 0 and 
                  close[i] < ema34 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or price below weekly EMA34
            if (trix_prev >= 0 and trix_val < 0) or close[i] < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or price above weekly EMA34
            if (trix_prev <= 0 and trix_val > 0) or close[i] > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals