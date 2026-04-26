#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1-week EMA trend filter and volume spike confirmation.
Long: price breaks above R1 + above 1w EMA50 + volume spike.
Short: price breaks below S1 + below 1w EMA50 + volume spike.
Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year).
Uses discrete position sizing (0.25) to limit fee drag. Works in bull/bear via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + 1.09*(high-low), S1 = close - 1.09*(high-low)
    camarilla_R1 = close_1d + 1.09 * (high_1d - low_1d)
    camarilla_S1 = close_1d - 1.09 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (completed 1d bars only)
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # 12h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA, 20 for volume MA)
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
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > R1_aligned[i]   # Price breaks above R1
        breakout_down = close[i] < S1_aligned[i]  # Price breaks below S1
        
        # 1w EMA50 trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if breakout_up and uptrend and volume_spike:
            # Long signal: break above R1 + uptrend + volume spike
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif breakout_down and downtrend and volume_spike:
            # Short signal: break below S1 + downtrend + volume spike
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

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0