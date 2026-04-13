#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation
Hypothesis: Use 4h/1d structure for signal direction (breakout above/below daily Camarilla R4/S4 with volume confirmation) and 1h for precise entry timing. Only trade during 08-20 UTC to reduce noise. Designed for 1h timeframe with controlled trade frequency (target 15-37 trades/year) to avoid fee drag. Works in both bull/bear markets via volatility expansion breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h data for additional confirmation (optional trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Previous day's values for today's calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate daily Camarilla levels
    range_1d = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r4 = camarilla_pp + (range_1d * 1.1 / 2)
    camarilla_s4 = camarilla_pp - (range_1d * 1.1 / 2)
    
    # Align 1d data to 1h
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume calculation: current day volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_condition_1d = vol_1d > (vol_ma_20 * 2.0)
    vol_condition_aligned = align_htf_to_ltf(prices, df_1d, vol_condition_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_condition_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during session
        if not session_filter[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = prices['close'].iloc[i] > camarilla_r4_aligned[i]
        short_breakout = prices['close'].iloc[i] < camarilla_s4_aligned[i]
        
        # Exit when price crosses daily pivot point
        long_exit = prices['close'].iloc[i] < camarilla_pp_aligned[i]
        short_exit = prices['close'].iloc[i] > camarilla_pp_aligned[i]
        
        if position == 0:
            if long_breakout and vol_condition_aligned[i]:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0