#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h momentum strategy using 1-day Donchian breakout with volume confirmation
# Long when price breaks above 1-day Donchian upper channel with volume > 1.5x 24-period average
# Short when price breaks below 1-day Donchian lower channel with same volume condition
# Exit when price crosses back inside the Donchian channel
# Uses 1-day Donchian channels for structural breakouts, volume to confirm conviction
# Designed for ~20-40 trades/year on major pairs
name = "6h_1dDonchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian channels (20-period) on 1d high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: 20-period high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current volume > 1.5x 24-period average (24 * 6h = 6 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume
            if close_val > upper and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume
            elif close_val < lower and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below upper Donchian
            if close_val < upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above lower Donchian
            if close_val > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals