#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Supertrend trend filter + 1d Camarilla R4/S4 breakout + volume confirmation.
Long when price breaks above daily Camarilla R4 level with 12h Supertrend uptrend and volume > 2.0x 20-period 4h volume average.
Short when price breaks below daily Camarilla S4 level with 12h Supertrend downtrend and volume > 2.0x 20-period 4h volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Supertrend filters whipsaws; Camarilla R4/S4 are strong breakout levels; volume confirms institutional participation.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation).
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
    
    # Get 4h data for volume average
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Supertrend (ATR=10, multiplier=3.0)
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr_12h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upperband = hl2 + (3.0 * atr_12h)
    lowerband = hl2 - (3.0 * atr_12h)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    start_idx = 10
    if len(close_12h) > start_idx:
        supertrend[start_idx] = upperband[start_idx]
        direction[start_idx] = 1  # start with uptrend
        
        for i in range(start_idx + 1, len(close_12h)):
            if close_12h[i] > supertrend[i-1]:
                direction[i] = 1
            elif close_12h[i] < supertrend[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1 and direction[i-1] == -1:
                supertrend[i] = lowerband[i]
            elif direction[i] == -1 and direction[i-1] == 1:
                supertrend[i] = upperband[i]
            elif direction[i] == 1:
                supertrend[i] = max(supertrend[i-1], lowerband[i])
            else:  # direction[i] == -1
                supertrend[i] = min(supertrend[i-1], upperband[i])
    
    # Calculate daily Camarilla levels
    # Camarilla: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align all to primary timeframe (4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_4h_aligned[i]
        # Trend filter: Supertrend direction
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        if position == 0:
            # Long: price breaks above daily Camarilla R4 with uptrend and volume
            if (close[i] > camarilla_r4_aligned[i] and 
                uptrend and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Camarilla S4 with downtrend and volume
            elif (close[i] < camarilla_s4_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily Camarilla R3 level
            camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1/4)
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily Camarilla S3 level
            camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1/4)
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hSupertrend_1dCamarilla_R4S4_Volume_Confirm"
timeframe = "4h"
leverage = 1.0