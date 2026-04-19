#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price breaks above/below 1d Donchian(10) channels with volume confirmation.
# Uses 1-day high/low channels to capture medium-term breakouts, filtered by volume > 1.5x 20-period average.
# Works in bull markets (breakouts up) and bear markets (breakdowns down). Target: 25-35 trades/year per symbol.
name = "12h_Donchian10_1d_Volume_Breakout"
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 10-period Donchian channels on daily
    donch_high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donch_low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align 1d Donchian channels to 12h
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_10)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_10)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure Donchian and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = donch_high_aligned[i]
        lower_channel = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above upper channel with volume
            if price > upper_channel and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below lower channel with volume
            elif price < lower_channel and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below lower channel
            if price < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above upper channel
            if price > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals