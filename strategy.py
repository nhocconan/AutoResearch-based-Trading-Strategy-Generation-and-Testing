#!/usr/bin/env python3
name = "1d_4Wk_Pivot_Support_Resistance_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA10 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=10, min_periods=10).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 4-week pivot levels from weekly data
    # Use 4-week lookback for pivot calculation (last 4 complete weeks)
    high_4w = []
    low_4w = []
    close_4w = []
    
    for i in range(len(df_1w)):
        if i >= 3:  # Need at least 4 weeks of data
            high_4w.append(np.max(df_1w['high'].values[i-3:i+1]))
            low_4w.append(np.min(df_1w['low'].values[i-3:i+1]))
            close_4w.append(df_1w['close'].values[i])
        else:
            high_4w.append(np.nan)
            low_4w.append(np.nan)
            close_4w.append(np.nan)
    
    high_4w = np.array(high_4w)
    low_4w = np.array(low_4w)
    close_4w = np.array(close_4w)
    
    # Calculate 4-week pivot points (using weekly OHLC)
    pivot_point = (high_4w + low_4w + close_4w) / 3.0
    # Resistance and support levels (using standard pivot formulas)
    r1 = 2 * pivot_point - low_4w
    s1 = 2 * pivot_point - high_4w
    
    # Align 4-week pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above S1 (support) AND above weekly EMA10 (uptrend) AND volume surge
            if close[i] > s1_aligned[i] and close[i] > ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 (resistance) AND below weekly EMA10 (downtrend) AND volume surge
            elif close[i] < r1_aligned[i] and close[i] < ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below pivot point OR below weekly EMA10 (trend change)
            if close[i] < pivot_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above pivot point OR above weekly EMA10 (trend change)
            if close[i] > pivot_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals