#!/usr/bin/env python3
# 4h_TRIX_13_Signal_12hTrend_Volume
# Hypothesis: TRIX (13) crossing zero with 12h EMA50 trend filter and volume spike for momentum entries.
# TRIX filters noise and identifies sustained momentum. Works in bull via zero-cross longs,
# in bear via zero-cross shorts. Volume confirms institutional interest. 12h trend avoids counter-trend trades.
# Target: 20-35 trades/year with strict entry conditions to minimize fee drag.

timeframe = "4h"
name = "4h_TRIX_13_Signal_12hTrend_Volume"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h closes
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate TRIX(13) on close: triple EMA then percent change
    ema1 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    ema2 = ema1.ewm(span=13, adjust=False, min_periods=13).mean()
    ema3 = ema2.ewm(span=13, adjust=False, min_periods=13).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values
    
    # Volume spike detection: 2x average volume (24-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(39, 24)  # TRIX needs 39 bars (13*3), vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and 12h uptrend
            if trix[i-1] <= 0 and trix[i] > 0 and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike and 12h downtrend
            elif trix[i-1] >= 0 and trix[i] < 0 and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or trend failure
            if trix[i] < 0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend failure
            if trix[i] > 0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals