#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR (14-period) for volatility filter
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly high/low for breakout levels (20-period)
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_1w_aligned[i]) or np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]):
            continue
        
        # Skip extremely low volatility periods (weekly ATR < 50% of its 50-period mean)
        if i >= 50:
            atr_mean_50 = np.nanmean(atr_1w_aligned[i-50:i])
            if not np.isnan(atr_mean_50) and atr_1w_aligned[i] < 0.5 * atr_mean_50:
                continue
        
        if position == 0:
            # Long: Break above 20-week high with expanding volatility
            if close[i] > high_20w_aligned[i] and atr_1w_aligned[i] > atr_1w_aligned[i-1]:
                position = 1
                signals[i] = position_size
            # Short: Break below 20-week low with expanding volatility
            elif close[i] < low_20w_aligned[i] and atr_1w_aligned[i] > atr_1w_aligned[i-1]:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Close below 20-week low or volatility contraction
            if close[i] < low_20w_aligned[i] or atr_1w_aligned[i] < 0.8 * atr_1w_aligned[i-1]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Close above 20-week high or volatility contraction
            if close[i] > high_20w_aligned[i] or atr_1w_aligned[i] < 0.8 * atr_1w_aligned[i-1]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1w_Volatility_Breakout_ATR_Filter_v1"
timeframe = "6h"
leverage = 1.0