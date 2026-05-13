#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_VolumeSpike_1dTrend_Filter
Hypothesis: TRIX (triple exponential average) crossing zero with volume confirmation
and 1-day trend alignment captures momentum reversals. Works in both bull and bear
markets by catching trend changes early. Volume filter reduces false signals.
Target: 20-40 trades/year to minimize fee drag.
"""

name = "4h_TRIX_ZeroCross_VolumeSpike_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX calculation: triple EMA of log returns
    close_series = pd.Series(close)
    log_returns = np.log(close_series / close_series.shift(1))
    ema1 = log_returns.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)  # percentage change
    trix_values = trix.fillna(0).values
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if position == 0:
            # LONG: TRIX crosses above zero with volume confirmation and uptrend
            if (trix_values[i-1] <= 0 and trix_values[i] > 0 and
                volume_filter[i] and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume confirmation and downtrend
            elif (trix_values[i-1] >= 0 and trix_values[i] < 0 and
                  volume_filter[i] and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend reverses
            if (trix_values[i] < 0) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend reverses
            if (trix_values[i] > 0) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals