#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Supertrend filter + 1d ADX regime filter + volume spike confirmation.
Long when: 12h Supertrend is bullish, 1d ADX > 25 (trending market), and 6h volume > 2.0x 20-period average.
Short when: 12h Supertrend is bearish, 1d ADX > 25, and 6h volume > 2.0x 20-period average.
Supertrend captures trend direction with dynamic ATR-based stops, ADX filters ranging markets,
volume spike confirms institutional participation. Target: 50-150 total trades over 4 years.
"""

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
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10)
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 12h Supertrend
    factor = 3.0
    hl2_12h = (high_12h + low_12h) / 2
    upperband_12h = hl2_12h + (factor * atr_12h)
    lowerband_12h = hl2_12h - (factor * atr_12h)
    
    supertrend_12h = np.full_like(close_12h, np.nan)
    direction_12h = np.full_like(close_12h, np.nan)  # 1 for up, -1 for down
    
    supertrend_12h[0] = hl2_12h[0]
    direction_12h[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upperband_12h[i-1]:
            direction_12h[i] = 1
        elif close_12h[i] < lowerband_12h[i-1]:
            direction_12h[i] = -1
        else:
            direction_12h[i] = direction_12h[i-1]
            if direction_12h[i] == 1 and lowerband_12h[i] < lowerband_12h[i-1]:
                lowerband_12h[i] = lowerband_12h[i-1]
            if direction_12h[i] == -1 and upperband_12h[i] > upperband_12h[i-1]:
                upperband_12h[i] = upperband_12h[i-1]
        
        if direction_12h[i] == 1:
            supertrend_12h[i] = lowerband_12h[i]
        else:
            supertrend_12h[i] = upperband_12h[i]
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d)
    dx = (np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di))) * 100
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)  # using 1d index for 6h volume MA
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for ADX and Supertrend
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_12h_aligned[i]) or np.isnan(direction_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: bullish Supertrend, ADX > 25 (trending), volume spike
            if (direction_12h_aligned[i] == 1 and 
                adx_1d_aligned[i] > 25 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: bearish Supertrend, ADX > 25 (trending), volume spike
            elif (direction_12h_aligned[i] == -1 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend turns bearish or ADX drops below 20 (ranging)
            if (direction_12h_aligned[i] == -1 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend turns bullish or ADX drops below 20 (ranging)
            if (direction_12h_aligned[i] == 1 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hSupertrend_1dADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0