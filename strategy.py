#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Filtered_v3
Hypothesis: Daily Camarilla R1/S1 breakouts with weekly EMA50 trend filter and volume spike confirmation.
Targets higher-quality trades by aligning with weekly trend, reducing whipsaws in counter-trend markets.
Works in both bull and bear via weekly trend filter. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to daily timeframe (completed daily bars only)
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Daily volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Camarilla R1/S1 breakout conditions
        breakout_up = close[i] > R1_aligned[i]   # Price breaks above R1
        breakout_down = close[i] < S1_aligned[i]  # Price breaks below S1
        
        # Weekly EMA50 trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if breakout_up and uptrend and volume_spike:
            # Long signal: break above R1 + weekly uptrend + volume spike
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif breakout_down and downtrend and volume_spike:
            # Short signal: break below S1 + weekly downtrend + volume spike
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Filtered_v3"
timeframe = "1d"
leverage = 1.0