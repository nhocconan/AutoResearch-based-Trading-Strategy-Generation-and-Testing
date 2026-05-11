#!/usr/bin/env python3
"""
1d_4H_Camarilla_R1_S1_Breakout_1wTrend_VolumeS_v1
Hypothesis: Daily chart strategy using weekly Camarilla R1/S1 breakout with 1-week EMA trend filter and volume spike confirmation.
Designed to work in both bull and bear markets by combining breakout momentum with trend alignment and volume confirmation.
Target: 30-100 total trades over 4 years on 1d timeframe.
"""

name = "1d_4H_Camarilla_R1_S1_Breakout_1wTrend_VolumeS_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # === 1W Data for Weekly Camarilla Pivots (previous week) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Calculate Weekly Camarilla levels for previous week
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to daily
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # === 1W Data for Trend Filter ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Volume Spike Filter (daily) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma20
    vol_spike = vol_ratio > 1.5  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike and weekly uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and ema20_1w_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike and weekly downtrend
            elif close[i] < S1_aligned[i] and vol_spike[i] and ema20_1w_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 or volume dries up
            if close[i] < S1_aligned[i] or vol_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above weekly R1 or volume dries up
            if close[i] > R1_aligned[i] or vol_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals