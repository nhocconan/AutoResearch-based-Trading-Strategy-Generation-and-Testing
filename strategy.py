#!/usr/bin/env python3
# 1h_4h_1d_camarilla_range_v1
# Strategy: 1h mean reversion with 4h/1d Camarilla levels and volume filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Price reverts to mean from extreme Camarilla levels (H3/L3) with volume confirmation.
# Works in both bull/bear markets by fading extremes. Uses 4h for direction filter, 1d for stronger S/R.
# Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_range_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Camarilla levels (based on previous day's range)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla for each 4h bar using prior 4h bar's range
    h3_4h = np.zeros_like(close_4h)
    l3_4h = np.zeros_like(close_4h)
    for i in range(1, len(close_4h)):
        range_ = high_4h[i-1] - low_4h[i-1]
        h3_4h[i] = close_4h[i-1] + range_ * 1.1 / 6
        l3_4h[i] = close_4h[i-1] - range_ * 1.1 / 6
    
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    
    # 1d Camarilla levels (more significant S/R)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    h3_1d = np.zeros_like(close_1d)
    l3_1d = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        range_ = high_1d[i-1] - low_1d[i-1]
        h3_1d[i] = close_1d[i-1] + range_ * 1.1 / 6
        l3_1d[i] = close_1d[i-1] - range_ * 1.1 / 6
    
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any data invalid
        if (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: above average
        vol_ok = volume[i] > vol_ma[i]
        
        # Mean reversion from extreme levels
        near_h3_4h = close[i] >= h3_4h_aligned[i] * 0.998  # Within 0.2% of H3
        near_l3_4h = close[i] <= l3_4h_aligned[i] * 1.002  # Within 0.2% of L3
        near_h3_1d = close[i] >= h3_1d_aligned[i] * 0.995  # Within 0.5% of H3 (stronger)
        near_l3_1d = close[i] <= l3_1d_aligned[i] * 1.005  # Within 0.5% of L3 (stronger)
        
        # Entry conditions: fade extremes with volume
        if vol_ok and near_h3_1d and position != -1:  # Strong rejection at 1d H3 -> short
            position = -1
            signals[i] = -0.20
        elif vol_ok and near_l3_1d and position != 1:  # Strong bounce at 1d L3 -> long
            position = 1
            signals[i] = 0.20
        elif vol_ok and near_h3_4h and not near_h3_1d and position != -1:  # Weak rejection at 4h H3 -> short
            position = -1
            signals[i] = -0.20
        elif vol_ok and near_l3_4h and not near_l3_1d and position != 1:  # Weak bounce at 4h L3 -> long
            position = 1
            signals[i] = 0.20
        # Exit conditions: return to mean or opposite signal
        elif position == 1 and (close[i] <= (h3_4h_aligned[i] + l3_4h_aligned[i]) / 2 or near_l3_4h):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= (h3_4h_aligned[i] + l3_4h_aligned[i]) / 2 or near_h3_4h):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals