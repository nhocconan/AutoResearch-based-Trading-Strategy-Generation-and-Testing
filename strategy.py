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
    
    # === 6h Donchian channel (20) ===
    highest_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # === 1d volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # === 6h volume confirmation ===
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current 1d volume > 1.5x 20-period average AND 6h volume > 1.3x 10-period average
        vol_spike_1d = volume[i] > vol_ma_20_aligned[i] * 1.5
        vol_spike_6h = volume[i] > vol_ma_10[i] * 1.3
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above previous high
        breakout_down = close[i] < lowest_low[i-1]  # break below previous low
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: upward breakout + volume spikes
            if breakout_up and vol_spike_1d and vol_spike_6h:
                signals[i] = 0.25
                position = 1
                continue
            # Short: downward breakout + volume spikes
            elif breakout_down and vol_spike_1d and vol_spike_6h:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price returns to midline or opposite breakout
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midline or breakout_down:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to midline or opposite breakout
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midline or breakout_up:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dVolume1.5x_6hVolume1.3x"
timeframe = "6h"
leverage = 1.0