#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and 1d volume confirmation
# In weekly uptrend (price > weekly EMA50): long breakouts above 6h Donchian H20 with volume spike
# In weekly downtrend (price < weekly EMA50): short breakouts below 6h Donchian L20 with volume spike
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Weekly EMA50 provides robust trend filter that works in both bull and bear markets
# Volume confirmation ensures breakouts have participation, reducing false signals

name = "6h_1w_1d_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    volume_1d_series = pd.Series(volume_1d)
    avg_volume_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d average volume to 6h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 x 1d average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian L20 or weekly trend turns down
            if close[i] < lowest_low[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian H20 or weekly trend turns up
            if close[i] > highest_high[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above Donchian H20 with volume confirmation in weekly uptrend
            if close[i] > highest_high[i] and volume_confirmed and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian L20 with volume confirmation in weekly downtrend
            elif close[i] < lowest_low[i] and volume_confirmed and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals