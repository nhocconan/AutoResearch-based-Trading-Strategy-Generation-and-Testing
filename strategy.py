#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 12h Donchian breakout and volume confirmation.
# Long when price breaks above 12h Donchian upper band with volume > 1.5x average.
# Short when price breaks below 12h Donchian lower band with volume > 1.5x average.
# Uses 12h timeframe for trend structure to avoid noise and reduce trade frequency.
# Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 20-period Donchian channels on 12h data
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    upper_band = align_htf_to_ltf(prices, df_12h, high_20)
    lower_band = align_htf_to_ltf(prices, df_12h, low_20)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper band with volume
            if price > upper_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower band with volume
            elif price < lower_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 12h Donchian lower band
            if price < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 12h Donchian upper band
            if price > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Donchian20_Breakout_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0