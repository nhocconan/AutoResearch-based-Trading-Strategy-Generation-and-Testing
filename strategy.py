#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ATR filter and volume confirmation
# Uses 12h Donchian channel breakout (20-period) with confirmation from 1d ATR and volume.
# Breakouts above upper band or below lower band are traded only when 1d ATR confirms volatility
# and volume is above average. Designed to capture trending moves while filtering false breakouts.
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 12h data for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channel on 12h
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    high_max_20_aligned = align_htf_to_ltf(prices, df_12h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_12h, low_min_20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(high_max_20_aligned[i]) or 
            np.isnan(low_min_20_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian upper band + ATR confirmation + volume confirmation
        if (close[i] > high_max_20_aligned[i] and
            atr_1d_aligned[i] > 0.5 * np.median(atr_1d_aligned[max(0, i-10):i+1]) and
            volume[i] > 1.5 * np.median(volume[max(0, i-10):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower band + ATR confirmation + volume confirmation
        elif (close[i] < low_min_20_aligned[i] and
              atr_1d_aligned[i] > 0.5 * np.median(atr_1d_aligned[max(0, i-10):i+1]) and
              volume[i] > 1.5 * np.median(volume[max(0, i-10):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ATR collapse (low volatility)
        elif position == 1 and (close[i] < low_min_20_aligned[i] or 
                                atr_1d_aligned[i] < 0.3 * np.median(atr_1d_aligned[max(0, i-10):i+1])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > high_max_20_aligned[i] or 
                                 atr_1d_aligned[i] < 0.3 * np.median(atr_1d_aligned[max(0, i-10):i+1])):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_ATR_Volume"
timeframe = "12h"
leverage = 1.0