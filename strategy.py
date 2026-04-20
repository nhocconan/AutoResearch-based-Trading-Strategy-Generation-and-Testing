#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 1d Donchian breakout and volume confirmation.
# Long when price breaks above previous 1d high with volume > 1.5x 20-period average.
# Short when price breaks below previous 1d low with volume > 1.5x 20-period average.
# Uses daily price extremes to avoid false breakouts and reduce trade frequency.
# Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's high and low (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Align daily high/low to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
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
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or np.isnan(vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        high_val = high_1d_aligned[i]
        low_val = low_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above previous day's high with volume
            if price > high_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous day's low with volume
            elif price < low_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below previous day's low
            if price < low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above previous day's high
            if price > high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian_Breakout_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0