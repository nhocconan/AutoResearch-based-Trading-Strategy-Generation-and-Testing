#!/usr/bin/env python3
# 4h_Trix_Volume_Trend_Filter
# Hypothesis: Use TRIX (1-period ROC of triple-smoothed EMA) to detect momentum on 4h, confirmed by 12h trend (EMA50) and volume spikes (>2x 20-period average). Enter long when TRIX crosses above zero and price > 12h EMA50 with volume spike; short when TRIX crosses below zero and price < 12h EMA50 with volume spike. Exit on TRIX zero-cross reverse. Targets 20-50 trades/year to minimize fee drift and work in both bull/bear via trend filter.

name = "4h_Trix_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # Calculate TRIX on 4h (15-period EMA applied 3 times + 1-period ROC)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change(1) * 100  # 1-period ROC as percentage
    trix_values = trix.values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(15, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + price > 12h EMA50 + volume spike
            if (trix_values[i-1] <= 0 and trix_values[i] > 0 and 
                close[i] > ema50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + price < 12h EMA50 + volume spike
            elif (trix_values[i-1] >= 0 and trix_values[i] < 0 and 
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix_values[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix_values[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals