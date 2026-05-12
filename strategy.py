#!/usr/bin/env python3
"""
1d_TRIX_ZeroCross_VolumeSpike_WeeklyTrend
Hypothesis: TRIX zero-cross signals with volume confirmation and weekly trend filter capture momentum shifts with low frequency. Works in bull markets via upward crosses and in bear markets via downward crosses, avoiding chop via weekly trend alignment. Target: 10-20 trades/year.
"""

name = "1d_TRIX_ZeroCross_VolumeSpike_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX: 15-period EMA applied 3 times, then 1-period ROC
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # 1-period ROC in percent
    trix_values = trix.values
    
    # Weekly trend: weekly EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: >2.0x 50-period average (infrequent)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(150, n):
        if (np.isnan(trix_values[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero + weekly uptrend + volume spike
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and
                close[i] > ema_50_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + weekly downtrend + volume spike
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and
                  close[i] < ema_50_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR weekly downtrend
            if (trix_values[i] < 0 and trix_values[i-1] >= 0) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR weekly uptrend
            if (trix_values[i] > 0 and trix_values[i-1] <= 0) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals