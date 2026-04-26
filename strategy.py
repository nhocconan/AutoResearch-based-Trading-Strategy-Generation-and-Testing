#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>2.0x median) to capture strong moves in both bull and bear markets. Enters long when price breaks above upper Donchian with volume confirmation and bullish 1d trend. Enters short when price breaks below lower Donchian with volume confirmation and bearish 1d trend. Exits on opposite breakout or when trend reverses. Uses discrete position sizing (0.25) to minimize churn. Target: 75-200 trades over 4 years. Works in bull markets by capturing breakouts and in bear markets by following 1d trend filter for short opportunities.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels for 4h (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    high_4h_prev[0] = np.nan
    low_4h_prev[0] = np.nan
    
    # Calculate Donchian(20) levels
    highest_high_20 = pd.Series(high_4h_prev).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h_prev).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h primary timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, highest_high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    
    # Volume confirmation: volume > 2.0x 50-period median (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median, 50-period EMA, 20-period Donchian)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above upper Donchian + volume confirmation + bullish 1d trend
        if close[i] > upper_20_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below lower Donchian + volume confirmation + bearish 1d trend
        elif close[i] < lower_20_aligned[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to the other level)
        elif position == 1 and close[i] < lower_20_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > upper_20_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Exit: trend reversal (price crosses EMA50 in opposite direction)
        elif position == 1 and close[i] < ema_50_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_50_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0