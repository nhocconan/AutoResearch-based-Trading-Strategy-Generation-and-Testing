#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Reversal_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot points from 1d data (pivot = (H+L+C)/3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from prior week (Friday close)
    # Using 5-day lookback for weekly pivot (approximation)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Align to 6h timeframe with proper delay for weekly pivot confirmation
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # 1d trend filter: EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near weekly pivot + uptrend + volume filter
            # Enter long when price is within 0.5% of pivot and trending up
            near_pivot_long = abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] < 0.005
            long_cond = near_pivot_long and (close[i] > ema_34_1d_aligned[i]) and volume_filter[i]
            
            # Short: price near weekly pivot + downtrend + volume filter
            near_pivot_short = abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] < 0.005
            short_cond = near_pivot_short and (close[i] < ema_34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves away from pivot or trend breaks
            if close[i] < ema_34_1d_aligned[i] or abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves away from pivot or trend breaks
            if close[i] > ema_34_1d_aligned[i] or abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals