#!/usr/bin/env python3
"""
12h_TRIX_ZeroCross_Volume_Trend
Hypothesis: TRIX(12) zero cross with 1w EMA50 trend filter and volume > 1.8x 20-period average
generates reliable signals on 12h timeframe. TRIX filters noise and zero crosses signal
momentum shifts. Volume surge confirms conviction. Works in bull (momentum continuation)
and bear (sharp reversals) by requiring alignment with 1w trend.
"""

name = "12h_TRIX_ZeroCross_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate TRIX(12): triple EMA of ROC
    # ROC = (close - close.lag(1)) / close.lag(1)
    close_series = pd.Series(close)
    roc = close_series.pct_change(1)
    # Triple EMA
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values * 100  # scale for readability

    # Volume confirmation: 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current 12h bar
        ema50 = ema50_1w_aligned[i]
        trix_val = trix[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(trix_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + price above 1w EMA50 + volume surge
            if i > 0 and trix[i-1] <= 0 and trix_val > 0:
                if close[i] > ema50 and volume[i] > vol_avg_val * 1.8:
                    signals[i] = 0.25
                    position = 1
            # SHORT: TRIX crosses below zero + price below 1w EMA50 + volume surge
            elif i > 0 and trix[i-1] >= 0 and trix_val < 0:
                if close[i] < ema50 and volume[i] > vol_avg_val * 1.8:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or price below 1w EMA50
            if i > 0 and trix[i-1] >= 0 and trix_val <= 0:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or price above 1w EMA50
            if i > 0 and trix[i-1] <= 0 and trix_val >= 0:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals