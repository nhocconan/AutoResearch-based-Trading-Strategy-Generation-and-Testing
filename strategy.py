#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Camarilla pivot levels from daily data act as strong support/resistance. Price breaking above/below these levels with volume expansion indicates institutional participation. This strategy works in bull markets (breakouts above resistance) and bear markets (breakdowns below support). Uses 12h timeframe to reduce trade frequency and avoid fee drag. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Formula: 
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # R1 = close + (high - low) * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.1/12
    # S2 = close - (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    
    cam_r4 = np.zeros_like(high_1d)
    cam_r3 = np.zeros_like(high_1d)
    cam_r2 = np.zeros_like(high_1d)
    cam_r1 = np.zeros_like(high_1d)
    cam_pp = np.zeros_like(high_1d)
    cam_s1 = np.zeros_like(high_1d)
    cam_s2 = np.zeros_like(high_1d)
    cam_s3 = np.zeros_like(high_1d)
    cam_s4 = np.zeros_like(high_1d)
    
    for i in range(len(df_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        range_hl = h - l
        
        cam_pp[i] = (h + l + c) / 3
        cam_r4[i] = c + range_hl * 1.1 / 2
        cam_r3[i] = c + range_hl * 1.1 / 4
        cam_r2[i] = c + range_hl * 1.1 / 6
        cam_r1[i] = c + range_hl * 1.1 / 12
        cam_s1[i] = c - range_hl * 1.1 / 12
        cam_s2[i] = c - range_hl * 1.1 / 6
        cam_s3[i] = c - range_hl * 1.1 / 4
        cam_s4[i] = c - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    cam_pp_aligned = align_htf_to_ltf(prices, df_1d, cam_pp)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_r2_aligned = align_htf_to_ltf(prices, df_1d, cam_r2)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    cam_s2_aligned = align_htf_to_ltf(prices, df_1d, cam_s2)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    # Volume confirmation: current volume > 1.8x 24-period average (on 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    volume_expansion = volume > (vol_ma_24 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(cam_pp_aligned[i]) or np.isnan(cam_r1_aligned[i]) or 
            np.isnan(cam_s1_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R1 with volume expansion
        long_breakout = close[i] > cam_r1_aligned[i] and volume_expansion[i]
        
        # Short condition: price breaks below S1 with volume expansion
        short_breakdown = close[i] < cam_s1_aligned[i] and volume_expansion[i]
        
        # Exit conditions: price returns to pivot point
        exit_long = position == 1 and close[i] < cam_pp_aligned[i]
        exit_short = position == -1 and close[i] > cam_pp_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakdown and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0