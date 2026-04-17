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
    
    # Get daily data for 1d ATR and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-day ATR on daily data for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Donchian channels (20-day)
    donch_high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20_1d)
    donch_low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20_1d)
    
    # Calculate 4-hour volume SMA for volume confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_20_1d_aligned[i]) or 
            np.isnan(donch_low_20_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average
        volume_filter = volume[i] > (1.5 * vol_sma_20[i])
        
        if position == 0:
            # Long: price breaks above daily Donchian high with volume
            if close[i] > donch_high_20_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low with volume
            elif close[i] < donch_low_20_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel
            midpoint = (donch_high_20_1d_aligned[i] + donch_low_20_1d_aligned[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel
            midpoint = (donch_high_20_1d_aligned[i] + donch_low_20_1d_aligned[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyDonchian20_Breakout_MidpointExit_VolumeFilter"
timeframe = "4h"
leverage = 1.0