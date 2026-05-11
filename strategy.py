#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Pivot_Breakout_MultiTF
Hypothesis: Combine daily and weekly Camarilla pivot levels with 12h price action to capture
multi-timeframe breakout opportunities. Long when price breaks above weekly R3 and daily R3
with volume confirmation. Short when price breaks below weekly S3 and daily S3 with volume
confirmation. Exit when price returns to opposing pivot level (daily S1 for longs, daily R1 for shorts).
Uses 12h timeframe to target 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Multi-timeframe confluence (daily + weekly) filters false breaks, improving win rate in both bull and bear markets.
"""

name = "12h_1d_1w_Camarilla_Pivot_Breakout_MultiTF"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily and weekly data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- Daily Camarilla Pivots (from previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC to calculate today's pivots
    camarilla_high_1d = np.full_like(close_1d, np.nan)
    camarilla_low_1d = np.full_like(close_1d, np.nan)
    camarilla_close_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        camarilla_high_1d[i] = high_1d[i-1]
        camarilla_low_1d[i] = low_1d[i-1]
        camarilla_close_1d[i] = close_1d[i-1]
    
    # Calculate Daily Camarilla levels
    R3_1d = camarilla_close_1d + ((camarilla_high_1d - camarilla_low_1d) * 1.2500)
    S3_1d = camarilla_close_1d - ((camarilla_high_1d - camarilla_low_1d) * 1.2500)
    R1_1d = camarilla_close_1d + ((camarilla_high_1d - camarilla_low_1d) * 1.0833)
    S1_1d = camarilla_close_1d - ((camarilla_high_1d - camarilla_low_1d) * 1.0833)
    
    # --- Weekly Camarilla Pivots (from previous week) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use previous week's OHLC to calculate current week's pivots
    camarilla_high_1w = np.full_like(close_1w, np.nan)
    camarilla_low_1w = np.full_like(close_1w, np.nan)
    camarilla_close_1w = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        camarilla_high_1w[i] = high_1w[i-1]
        camarilla_low_1w[i] = low_1w[i-1]
        camarilla_close_1w[i] = close_1w[i-1]
    
    # Calculate Weekly Camarilla levels
    R3_1w = camarilla_close_1w + ((camarilla_high_1w - camarilla_low_1w) * 1.2500)
    S3_1w = camarilla_close_1w - ((camarilla_high_1w - camarilla_low_1w) * 1.2500)
    
    # Align pivots to 12h timeframe
    R3_1d_12h = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_12h = align_htf_to_ltf(prices, df_1d, S3_1d)
    R1_1d_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    R3_1w_12h = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_12h = align_htf_to_ltf(prices, df_1w, S3_1w)
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for volume MA and data alignment
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_1d_12h[i]) or np.isnan(S3_1d_12h[i]) or 
            np.isnan(R3_1w_12h[i]) or np.isnan(S3_1w_12h[i]) or
            np.isnan(R1_1d_12h[i]) or np.isnan(S1_1d_12h[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only with multi-timeframe confluence and volume
            if (close_12h[i] > R3_1d_12h[i] and close_12h[i] > R3_1w_12h[i] and vol_ok):
                # Long: price breaks above both daily and weekly R3 + volume
                signals[i] = 0.25
                position = 1
            elif (close_12h[i] < S3_1d_12h[i] and close_12h[i] < S3_1w_12h[i] and vol_ok):
                # Short: price breaks below both daily and weekly S3 + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to daily S1 (opposite side)
                if close_12h[i] <= S1_1d_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to daily R1 (opposite side)
                if close_12h[i] >= R1_1d_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals