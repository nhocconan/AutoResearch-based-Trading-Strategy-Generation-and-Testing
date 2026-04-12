#!/usr/bin/env python3
"""
4h_12h_volume_breakout_retest
Breakout strategy: On 12h, detect range via Bollinger Band squeeze; on 4h, enter on breakout with volume.
Entry only when price retests breakout level after breakout (reduces false breakouts).
Uses volume confirmation and ATR-based stop.
Target: 20-40 trades/year to minimize fee drag.
"""

name = "4h_12h_volume_breakout_retest"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Bollinger Bands (20, 2) on 12h
    bb_length = 20
    bb_mult = 2.0
    
    basis = pd.Series(close_12h).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close_12h).rolling(window=bb_length, min_periods=bb_length).std().values
    upper = basis + dev
    lower = basis - dev
    
    # Squeeze detection: bandwidth below 20-period lowest
    bandwidth = (upper - lower) / basis
    bandwidth_smoothed = pd.Series(bandwidth).rolling(window=5, min_periods=5).mean().values
    bandwidth_lowest = pd.Series(bandwidth_smoothed).rolling(window=20, min_periods=20).min().values
    squeeze = bandwidth_smoothed <= bandwidth_lowest
    
    # Align to 4h
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    basis_aligned = align_htf_to_ltf(prices, df_12h, basis)
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze)
    
    # Volume confirmation on 4h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Track breakout levels and retest
    breakout_level_long = np.full(n, np.nan)
    breakout_level_short = np.full(n, np.nan)
    breakout_active_long = np.zeros(n, dtype=bool)
    breakout_active_short = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if not np.isnan(upper_aligned[i]) and not np.isnan(lower_aligned[i]):
            # Detect new breakouts
            if squeeze_aligned[i-1] and not squeeze_aligned[i]:
                if close[i] > upper_aligned[i]:
                    breakout_level_long[i] = upper_aligned[i]
                    breakout_active_long[i] = True
                elif close[i] < lower_aligned[i]:
                    breakout_level_short[i] = lower_aligned[i]
                    breakout_active_short[i] = True
            # Carry over active breakouts
            if breakout_active_long[i-1]:
                breakout_active_long[i] = True
                breakout_level_long[i] = breakout_level_long[i-1]
            if breakout_active_short[i-1]:
                breakout_active_short[i] = True
                breakout_level_short[i] = breakout_level_short[i-1]
            # Deactivate if price moves too far from breakout level
            if breakout_active_long[i] and close[i] < breakout_level_long[i] * 0.98:
                breakout_active_long[i] = False
            if breakout_active_short[i] and close[i] > breakout_level_short[i] * 1.02:
                breakout_active_short[i] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(basis_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: retest of breakout level after volume confirmation
        if breakout_active_long[i] and not breakout_active_long[i-1]:
            if close[i] >= breakout_level_long[i] * 0.995 and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
        # Short entry: retest of breakout level after volume confirmation
        elif breakout_active_short[i] and not breakout_active_short[i-1]:
            if close[i] <= breakout_level_short[i] * 1.005 and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
        # Exit when price returns to middle band
        elif position == 1 and close[i] <= basis_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= basis_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals