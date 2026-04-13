#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume
Hypothesis: Uses Camarilla pivot levels from daily timeframe for breakout entries on 4h.
In ranging markets, price tends to revert to mean (pivot); in trending markets, breaks of
S3/S4 or R3/R4 levels indicate strong momentum. Volume confirmation filters false breaks.
Works in both bull and bear markets by trading breakouts of key intraday support/resistance.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
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
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    rng = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = close_1d + (rng * 1.1 / 12)
    camarilla_r2 = close_1d + (rng * 1.1 / 6)
    camarilla_r3 = close_1d + (rng * 1.1 / 4)
    camarilla_r4 = close_1d + (rng * 1.1 / 2)
    camarilla_s1 = close_1d - (rng * 1.1 / 12)
    camarilla_s2 = close_1d - (rng * 1.1 / 6)
    camarilla_s3 = close_1d - (rng * 1.1 / 4)
    camarilla_s4 = close_1d - (rng * 1.1 / 2)
    
    # Use R3/R4 and S3/S4 for breakouts (stronger levels)
    breakout_levels_up = np.maximum(camarilla_r3, camarilla_r4)
    breakout_levels_down = np.minimum(camarilla_s3, camarilla_s4)
    
    # Get 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_expansion_4h = volume_4h > (vol_ma_20_4h * 1.5)
    
    # Breakout conditions
    breakout_up = (close_4h > breakout_levels_up) & volume_expansion_4h
    breakout_down = (close_4h < breakout_levels_down) & volume_expansion_4h
    
    # Align all signals to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    breakout_up_aligned = align_htf_to_ltf(prices, df_4h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_4h, breakout_down)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(camarilla_pp_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(breakout_up_aligned[i]) or \
           np.isnan(breakout_down_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
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
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0