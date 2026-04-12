#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_donchian_breakout_volume_v1
# Donchian(20) breakout from daily chart with volume confirmation on 12h timeframe.
# Uses 1-day Donchian channels to capture major breakouts in both bull and bear markets.
# Volume confirmation ensures institutional participation, reducing false breakouts.
# Designed for low trade frequency (15-25/year) to minimize fee drag.
# Works in trending markets by riding breakouts and avoids chop via volume filter.
name = "12h_1d_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels from daily high/low
    # Use previous day's data to avoid look-ahead
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align to 12h timeframe (already delayed by 1 day due to shift)
    upper_band = align_htf_to_ltf(prices, df_1d, high_20)
    lower_band = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if bands not ready
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above upper Donchian band with volume
        if close[i] > upper_band[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below lower Donchian band with volume
        elif close[i] < lower_band[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < lower_band[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > upper_band[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals