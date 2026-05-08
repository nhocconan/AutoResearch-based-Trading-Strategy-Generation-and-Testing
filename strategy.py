#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA20 trend filter and volume spike confirmation
# Designed for 7-25 trades/year with proper risk control via trend failure
# Long: price breaks above 1d Donchian upper + price > 1w EMA20 + volume spike
# Short: price breaks below 1d Donchian lower + price < 1w EMA20 + volume spike
# Exit: trend failure (price crosses 1w EMA20) or opposite breakout
# Volume filter: current 1d volume > 1.5x 20-day average to avoid false breakouts
# Donchian channels provide clear trend-following structure, EMA20 on weekly filters trend, volume confirms breakout strength

name = "1d_Donchian_Breakout_1wEMA20_VolumeFilter"
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
    
    # Get 1d data for Donchian calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) from 1d data
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 1w and 1d indicators to 1d timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-day average
        vol_filter = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for breakout with trend and volume confirmation
            # Long: price breaks above Donchian upper + uptrend + volume spike
            if close[i] > high_20_aligned[i] and ema20_1w_aligned[i] > close[i]:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian lower + downtrend + volume spike
            elif close[i] < low_20_aligned[i] and ema20_1w_aligned[i] < close[i]:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend failure (price crosses below EMA20) or opposite breakout
            if ema20_1w_aligned[i] <= close[i] or close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend failure (price crosses above EMA20) or opposite breakout
            if ema20_1w_aligned[i] >= close[i] or close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals