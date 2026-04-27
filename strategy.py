#!/usr/bin/env python3
"""
12h_Trix_ZeroCross_Volume_Spike_1dTrend
Hypothesis: TRIX (1-period rate of change of triple EMA) crossing zero indicates momentum shift. Combined with volume spike and daily EMA trend filter, this captures strong momentum moves. Works in bull (zero cross up with volume) and bear (zero cross down with volume) by requiring volume confirmation. Target 15-25 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # TRIX: 1-period ROC of triple EMA (15-period as common setting)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix = pd.Series(ema3).pct_change() * 100  # 1-period ROC in percentage
    trix_values = trix.values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for TRIX (3*15 + 1), EMA, and volume MA
    start_idx = max(46, 20)  # 15*3 + 1 = 46 for TRIX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        trix_now = trix_values[i]
        trix_prev = trix_values[i-1]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + uptrend (price > EMA34)
            if trix_prev <= 0 and trix_now > 0 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: TRIX crosses below zero + volume spike + downtrend (price < EMA34)
            elif trix_prev >= 0 and trix_now < 0 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TRIX crosses below zero or trend turns down
            if trix_prev >= 0 and trix_now < 0 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TRIX crosses above zero or trend turns up
            if trix_prev <= 0 and trix_now > 0 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Trix_ZeroCross_Volume_Spike_1dTrend"
timeframe = "12h"
leverage = 1.0