#!/usr/bin/env python3
# 6h_Pivot_Reversion_1wTrend_Volume
# Hypothesis: Fade at daily pivot (PP) with volume confirmation when weekly trend is strong.
# In strong weekly trends, price tends to revert to daily pivot before continuing.
# Works in bull/bear: uses weekly trend filter, avoids counter-trend trades.
# Target: 15-30 trades/year (60-120 total over 4 years) with strict entry.
# Uses mean reversion at pivot + trend filter to reduce false signals.

timeframe = "6h"
name = "6h_Pivot_Reversion_1wTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Daily pivot point: (H + L + C) / 3
    pivot = (d_high + d_low + d_close) / 3.0
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume spike: 1.5x average volume (6-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(6, 50)  # Ensure we have volume MA and weekly EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below pivot in weekly uptrend with volume
            if close[i] < pivot_aligned[i] and ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above pivot in weekly downtrend with volume
            elif close[i] > pivot_aligned[i] and ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses above pivot or weekly trend fails
            if close[i] > pivot_aligned[i] or ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses below pivot or weekly trend fails
            if close[i] < pivot_aligned[i] or ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals