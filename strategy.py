#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume
Hypothesis: Combines daily Camarilla pivot levels with 12h breakout confirmation and volume filtering.
In trending markets, price breaks above/below key pivot levels (H4/L4) with volume > 1.5x 20-period average.
Works in both bull and bear markets by trading breakouts from key intraday support/resistance.
Target: 15-35 trades/year on 12h (60-140 total over 4 years).
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels for daily
    # Pivot = (H + L + C) / 3
    # H4 = Pivot + 1.5 * (H - L)
    # L4 = Pivot - 1.5 * (H - L)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    h4 = pivot + 1.5 * (high_1d - low_1d)
    l4 = pivot - 1.5 * (high_1d - low_1d)
    
    # Get 12h data for breakout confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period volume average on 12h
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean()
    volume_expansion_12h = volume_12h > (vol_ma_20_12h * 1.5)
    
    # 12h breakout conditions: close above H4 or below L4 with volume expansion
    breakout_up = (close_12h > h4) & volume_expansion_12h
    breakout_down = (close_12h < l4) & volume_expansion_12h
    
    # Align all signals to 12h timeframe (already aligned since using 12h data)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    breakout_up_aligned = align_htf_to_ltf(prices, df_12h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_12h, breakout_down)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(h4_aligned[i]) or \
           np.isnan(l4_aligned[i]) or \
           np.isnan(breakout_up_aligned[i]) or \
           np.isnan(breakout_down_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price breaks pivot level with volume confirmation
        if breakout_up_aligned[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif breakout_down_aligned[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Exit conditions: price returns to pivot area (between H4 and L4)
        elif position == 1 and close[i] < h4_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > l4_aligned[i]:
            position = 0
            signals[i] = 0.0
        # Hold position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0