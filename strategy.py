#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend (ATR=10, mult=3) with 1d ADX filter and 4h volume confirmation.
# Long when Supertrend turns green, 1d ADX > 25 (trending), and 4h volume > 2x 20-period average.
# Short when Supertrend turns red, 1d ADX > 25, and 4h volume > 2x 20-period average.
# Exit when Supertrend reverses (opposite color).
# Uses Supertrend for trend following, ADX to avoid ranging markets, volume to confirm strength.
# Target: 80-160 total trades over 4 years (20-40/year) for low fee drag.

name = "4h_Supertrend_1dADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR (10-period) for Supertrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = low[0] - close[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + (3 * atr)
    lowerband = hl2 - (3 * atr)
    
    # Initialize Supertrend arrays
    supetrend = np.zeros(n)
    supetrend_dir = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i] > supetrend[i-1]:
            supetrend[i] = lowerband[i]
            supetrend_dir[i] = 1
        else:
            supetrend[i] = upperband[i]
            supetrend_dir[i] = -1
        
        # Adjust bands
        if supetrend[i] < supetrend[i-1]:
            supetrend[i] = supetrend[i-1]
        if close[i] > supetrend[i]:
            supetrend[i] = lowerband[i]
        if close[i] < supetrend[i]:
            supetrend[i] = upperband[i]
    
    # 1d ADX (14-period) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = high_1d[0] - low_1d[0]
    tr2_1d[0] = high_1d[0] - close_1d[0]
    tr3_1d[0] = low_1d[0] - close_1d[0]
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(supetrend[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Supertrend uptrend, ADX > 25, volume spike
            long_cond = (supetrend_dir[i] == 1) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: Supertrend downtrend, ADX > 25, volume spike
            short_cond = (supetrend_dir[i] == -1) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.30
                position = 1
            elif short_cond:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Supertrend turns down
            if supetrend_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Supertrend turns up
            if supetrend_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals