#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 1d Donchian channel breakout and 1h volume confirmation.
# Long when price breaks above 1d Donchian upper (20-period high) with 1h volume > 1.5x average.
# Short when price breaks below 1d Donchian lower (20-period low) with 1h volume > 1.5x average.
# Uses 1d Donchian for structure, 1h volume for confirmation, and 4h price action for entry timing.
# Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper = 20-period high, Lower = 20-period low
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (wait for daily close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Load 4h data for price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume confirmation
            if price > upper and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume confirmation
            elif price < lower and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian20_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0