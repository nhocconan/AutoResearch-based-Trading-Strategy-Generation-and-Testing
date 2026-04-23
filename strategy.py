#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above Donchian(20) high AND weekly pivot > previous close AND volume > 2.0x 20-period MA.
Short when price breaks below Donchian(20) low AND weekly pivot < previous close AND volume > 2.0x 20-period MA.
Exit when price returns to Donchian(20) midpoint or weekly pivot direction reverses.
Uses 1w HTF for structural bias to avoid counter-trend trades in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Donchian channels provide objective breakout levels, weekly pivot adds higher-timeframe context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1w pivot points for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: P = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian (needs 20), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 6h volume > 2.0x 20-period MA (high threshold to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian high AND weekly pivot > previous close AND volume filter
            if (close[i] > donchian_high[i] and 
                pivot_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else False and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND weekly pivot < previous close AND volume filter
            elif (close[i] < donchian_low[i] and 
                  pivot_1w_aligned[i] < close_1w[-1] if len(close_1w) > 0 else False and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Price returns to Donchian midpoint OR weekly pivot direction reverses
                if (close[i] <= donchian_mid[i] or 
                    (len(close_1w) > 0 and pivot_1w_aligned[i] <= close_1w[-1])):
                    exit_signal = True
            elif position == -1:
                # Short exit: Price returns to Donchian midpoint OR weekly pivot direction reverses
                if (close[i] >= donchian_mid[i] or 
                    (len(close_1w) > 0 and pivot_1w_aligned[i] >= close_1w[-1])):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_1wPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0