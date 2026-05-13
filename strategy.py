#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_1dTrend
# Hypothesis: TRIX (15-period) crosses zero with volume spike and 1d EMA50 trend confirmation. 
# In uptrend (price > 1d EMA50), long on TRIX crossing above zero with volume > 1.5x 20-period average. 
# In downtrend (price < 1d EMA50), short on TRIX crossing below zero with volume spike. 
# Exits on TRIX crossing back through zero or trend reversal. 
# Designed to capture momentum in both bull and bear markets with low trade frequency.

name = "4h_TRIX_VolumeSpike_1dTrend"
timeframe = "4h"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate TRIX(15): EMA of EMA of EMA of price
    ema1 = pd.Series(close).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.fillna(0).values

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(trix_values[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero (bullish momentum) + price above 1d EMA50 (uptrend) + volume spike
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero (bearish momentum) + price below 1d EMA50 (downtrend) + volume spike
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero (loss of momentum) or price below 1d EMA50 (trend change)
            if (trix_values[i] < 0 and trix_values[i-1] >= 0) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero (loss of momentum) or price above 1d EMA50 (trend change)
            if (trix_values[i] > 0 and trix_values[i-1] <= 0) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals