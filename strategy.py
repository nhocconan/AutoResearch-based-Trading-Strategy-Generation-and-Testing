#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_20_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w Donchian(20) channel ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian bands with proper lookback
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # === 1d EMA50 trend filter ===
    ema50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume spike filter (1d) ===
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure Donchian and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20w_aligned[i]) or 
            np.isnan(low_20w_aligned[i]) or
            np.isnan(ema50_1d[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above weekly Donchian high + above EMA50 + volume spike
            if (close[i] > high_20w_aligned[i] and
                close[i] > ema50_1d[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian low + below EMA50 + volume spike
            elif (close[i] < low_20w_aligned[i] and
                  close[i] < ema50_1d[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below weekly Donchian low or below EMA50
            if close[i] < low_20w_aligned[i] or close[i] < ema50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above weekly Donchian high or above EMA50
            if close[i] > high_20w_aligned[i] or close[i] > ema50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals