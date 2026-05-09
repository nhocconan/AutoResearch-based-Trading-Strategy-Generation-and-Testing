#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_MomentumBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for momentum and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Momentum: 10-period ROC (Rate of Change) on 1d close
    close_1d = df_1d['close'].values
    roc_10 = np.full_like(close_1d, np.nan)
    roc_10[10:] = (close_1d[10:] - close_1d[:-10]) / close_1d[:-10] * 100
    
    # Volume filter: current 1d volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align all to 6h
    roc_10_6h = align_htf_to_ltf(prices, df_1d, roc_10)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need ROC and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(roc_10_6h[i]) or np.isnan(volume_filter_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        roc_val = roc_10_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: positive momentum + volume expansion
            if roc_val > 1.0 and vol_filter:  # >1% 10-day ROC
                signals[i] = 0.25
                position = 1
            # Enter short: negative momentum + volume expansion
            elif roc_val < -1.0 and vol_filter:  # <-1% 10-day ROC
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: momentum turns negative
            if roc_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: momentum turns positive
            if roc_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals