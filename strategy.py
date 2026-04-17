#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Supertrend trend filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above Donchian upper band with volume > 1.5x 20-period average and 12h Supertrend = uptrend.
Short when price breaks below Donchian lower band with volume > 1.5x 20-period average and 12h Supertrend = downtrend.
Exit on opposite Donchian band touch or trend reversal.
Uses ATR-based stoploss equivalent via close-based exit.
Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Calculate ATR for Supertrend (using 10-period ATR, multiplier 3.0)
    atr_period = 10
    atr_mult = 3.0
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    basic_ub = hl2 + atr_mult * atr
    basic_lb = hl2 - atr_mult * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = basic_ub[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = basic_lb[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = basic_ub[i]
        elif direction[i] == 1:
            supertrend[i] = max(basic_lb[i], supertrend[i-1])
        else:
            supertrend[i] = min(basic_ub[i], supertrend[i-1])
    
    # Get 12h data for Supertrend trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR for Supertrend
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = 0
    tr2_12h[0] = 0
    tr3_12h[0] = 0
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 12h basic upper and lower bands
    hl2_12h = (high_12h + low_12h) / 2
    basic_ub_12h = hl2_12h + atr_mult * atr_12h
    basic_lb_12h = hl2_12h - atr_mult * atr_12h
    
    # Initialize 12h Supertrend
    supertrend_12h = np.zeros_like(close_12h)
    direction_12h = np.ones_like(close_12h)
    
    supertrend_12h[0] = basic_ub_12h[0]
    direction_12h[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend_12h[i-1]:
            direction_12h[i] = 1
        elif close_12h[i] < supertrend_12h[i-1]:
            direction_12h[i] = -1
        else:
            direction_12h[i] = direction_12h[i-1]
        
        if direction_12h[i] == 1 and direction_12h[i-1] == -1:
            supertrend_12h[i] = basic_lb_12h[i]
        elif direction_12h[i] == -1 and direction_12h[i-1] == 1:
            supertrend_12h[i] = basic_ub_12h[i]
        elif direction_12h[i] == 1:
            supertrend_12h[i] = max(basic_lb_12h[i], supertrend_12h[i-1])
        else:
            supertrend_12h[i] = min(basic_ub_12h[i], supertrend_12h[i-1])
    
    # Align 12h Supertrend direction to 4h
    supertrend_12h_dir_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(donchian_period, 20)  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(supertrend_12h_dir_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volume and bullish 12h trend
            if (close[i] > highest_high[i] and 
                volume_confirmed and 
                supertrend_12h_dir_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band with volume and bearish 12h trend
            elif (close[i] < lowest_low[i] and 
                  volume_confirmed and 
                  supertrend_12h_dir_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches Donchian lower band or 12h trend turns bearish
            if (close[i] < lowest_low[i] or 
                supertrend_12h_dir_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches Donchian upper band or 12h trend turns bullish
            if (close[i] > highest_high[i] or 
                supertrend_12h_dir_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hSupertrend_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0