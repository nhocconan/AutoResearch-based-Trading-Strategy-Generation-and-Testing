#!/usr/bin/env python3
# 1h_4d_Donchian_Breakout_Trend_VolumeFilter
# Hypothesis: On 1h timeframe, trade breakouts of 4h Donchian channels with volume confirmation and 1d trend filter.
# In bull markets, price breaks above upper channel; in bear markets, breaks below lower channel.
# Uses 1d EMA50 for trend direction to avoid counter-trend trades.
# Targets 15-35 trades/year by requiring volume > 1.5x average and clear breakout.
# Uses 4h timeframe for signal direction, 1h only for entry timing precision.

name = "1h_4d_Donchian_Breakout_Trend_VolumeFilter"
timeframe = "1h"
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
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower bands
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above upper Donchian with volume confirmation and uptrend
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_aligned[i] and  # Uptrend filter
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short breakdown below lower Donchian with volume confirmation and downtrend
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_aligned[i] and  # Downtrend filter
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price returns to lower Donchian or trend reverses
            if close[i] < lower_aligned[i] or close[i] < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns to upper Donchian or trend reverses
            if close[i] > upper_aligned[i] or close[i] > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals