#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Pivot_HighLow_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard)
    # Pivot = (H + L + C)/3
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Range = H - L
    range_ = high_1w - low_1w
    # Support and Resistance levels
    r1 = pivot + range_
    s1 = pivot - range_
    r2 = pivot + 2 * range_
    s2 = pivot - 2 * range_
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily trend filter: EMA(50) on close
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R2 with daily uptrend and volume
            if (close[i] > r2_aligned[i] and 
                close[i] > ema_50[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with daily downtrend and volume
            elif (close[i] < s2_aligned[i] and 
                  close[i] < ema_50[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below pivot or reverses below EMA
            if (close[i] < pivot_aligned[i] or 
                close[i] < ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above pivot or reverses above EMA
            if (close[i] > pivot_aligned[i] or 
                close[i] > ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals