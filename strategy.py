#!/usr/bin/env python3
# 12h_TRIX_VolumeSpike_Trend
# Hypothesis: TRIX momentum combined with volume spike and 1w trend filter on 12h timeframe.
# Long when TRIX crosses above zero with volume spike and 1w uptrend.
# Short when TRIX crosses below zero with volume spike and 1w downtrend.
# TRIX filters noise and catches momentum shifts; volume confirms institutional participation.
# Works in bull markets (TRIX > 0 in uptrend) and bear markets (TRIX < 0 in downtrend).
# Target: 12-37 trades/year per symbol to minimize fee drag.

name = "12h_TRIX_VolumeSpike_Trend"
timeframe = "12h"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate TRIX on 12h close: EMA of EMA of EMA (15-period)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())  # percentage change of triple EMA
    trix_values = trix.values
    
    # 1w trend: EMA50
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w trend to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 2.0 * 3-period average (1.5 days worth at 12h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_values[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume spike + 1w uptrend
            if trix_values[i] > 0 and trix_values[i-1] <= 0 and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike + 1w downtrend
            elif trix_values[i] < 0 and trix_values[i-1] >= 0 and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend reversal
            if trix_values[i] < 0 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend reversal
            if trix_values[i] > 0 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals