#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for structure and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend (30 min periods)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Donchian channels (20-period) - breakout levels
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align daily levels to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donch_high = align_htf_to_ltf(prices, df_1d, highest_20)
    donch_low = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above Donchian high, above EMA50, volume spike
        if (close[i] > donch_high[i] and 
            close[i] > ema50_aligned[i] and 
            volume_spike[i]):
            signals[i] = 0.25
            position = 1
        # Short: price breaks below Donchian low, below EMA50, volume spike
        elif (close[i] < donch_low[i] and 
              close[i] < ema50_aligned[i] and 
              volume_spike[i]):
            signals[i] = -0.25
            position = -1
        # Exit: price returns to opposite Donchian level
        elif position == 1 and close[i] < donch_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donch_high[i]:
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

name = "4h_Donchian20_EMA50_Volume2x_1d_v1"
timeframe = "4h"
leverage = 1.0