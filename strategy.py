#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_1dVolume_S
Hypothesis: Use 1-day trend (EMA34) and volume spike (1.5x 20-period average) to filter 12-hour Donchian(20) breakouts.
Long when price breaks above 12h Donchian upper channel and 1d EMA34 is rising with volume confirmation.
Short when price breaks below 12h Donchian lower channel and 1d EMA34 is falling with volume confirmation.
Exit when price returns to the 12h Donchian middle (mean-reversion exit).
Designed to capture trends with filtered breakouts, reducing false signals in both bull and bear markets.
"""
name = "12h_Donchian20_Breakout_1dTrend_1dVolume_S"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    # Donchian upper: highest high over last 20 periods
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low over last 20 periods
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Donchian middle: average of upper and lower
    middle_12h = (upper_12h + lower_12h) / 2
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_12h_aligned = align_htf_to_ltf(prices, df_12h, middle_12h)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 1d data for volume filter
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(middle_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with 1d uptrend and volume spike
            if (high[i] > upper_12h_aligned[i] and 
                ema_1d_aligned[i] > ema_1d_aligned[i-1] and  # EMA rising
                volume[i] > vol_avg_1d_aligned[i] * 1.5):   # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with 1d downtrend and volume spike
            elif (low[i] < lower_12h_aligned[i] and 
                  ema_1d_aligned[i] < ema_1d_aligned[i-1] and  # EMA falling
                  volume[i] > vol_avg_1d_aligned[i] * 1.5):   # Volume spike
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to Donchian middle (mean reversion)
            if position == 1 and low[i] < middle_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and high[i] > middle_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals