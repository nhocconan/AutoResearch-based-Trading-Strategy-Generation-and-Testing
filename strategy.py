#1/1/1900 00:00:00
#!/usr/bin/env python3
# 6h_MACD_Histogram_Trend_Filter_1d
# Hypothesis: MACD histogram crossing zero with 1-day trend filter and volume confirmation
# captures momentum in both bull and bear markets. The MACD histogram (MACD - signal line)
# provides early trend change signals, while the 1-day EMA50 filters for higher timeframe
# trend direction to avoid counter-trend trades. Volume confirmation ensures breakouts
# have participation. Target: 15-35 trades per year per symbol to minimize fee drag.

name = "6h_MACD_Histogram_Trend_Filter_1d"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # MACD components
    fast = 12
    slow = 26
    signal = 9
    
    # EMA calculations
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    macd_hist = macd_line - macd_signal

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(macd_hist[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: MACD hist crosses above zero + 1d EMA50 uptrend + volume spike
            if (macd_hist[i] > 0 and macd_hist[i-1] <= 0 and  # crossed above zero
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: MACD hist crosses below zero + 1d EMA50 downtrend + volume spike
            elif (macd_hist[i] < 0 and macd_hist[i-1] >= 0 and  # crossed below zero
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: MACD hist crosses below zero or trend change
            if macd_hist[i] < 0 and macd_hist[i-1] >= 0:  # crossed below zero
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: MACD hist crosses above zero or trend change
            if macd_hist[i] > 0 and macd_hist[i-1] <= 0:  # crossed above zero
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals