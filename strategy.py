#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1d Donchian channel breakout (20-period) and volume confirmation.
# Long when price breaks above upper Donchian band with volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian band with volume > 1.5x 20-period average.
# Uses 1d Donchian to capture multi-day breakouts, avoiding false signals in sideways markets.
# Volume filter ensures breakouts have conviction. Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper band: 20-period high
    # Lower band: 20-period low
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe (shifted by 1 day to avoid look-ahead)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume
            if price > upper and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume
            elif price < lower and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian20_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0