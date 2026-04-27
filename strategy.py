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
    
    # Get weekly data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA for weekly trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily close prices for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian channels (20-period)
    highest_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average (daily)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_1d * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(highest_high_1d[i]) or 
            np.isnan(lowest_low_1d[i]) or np.isnan(vol_ma_1d[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Get daily close for today (need to access from 1d data)
        # Since we're on 1d timeframe, we can use close[i] directly
        daily_close = close[i]
        
        # Long conditions: price breaks above upper Donchian + weekly trend up + volume spike
        long_breakout = (daily_close > highest_high_1d[i-1] and 
                        daily_close > ema20_1w_aligned[i] and 
                        volume_spike[i])
        
        # Short conditions: price breaks below lower Donchian + weekly trend down + volume spike
        short_breakout = (daily_close < lowest_low_1d[i-1] and 
                         daily_close < ema20_1w_aligned[i] and 
                         volume_spike[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout with volume
        elif position == 1 and daily_close < lowest_low_1d[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and daily_close > highest_high_1d[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0