#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike_v1
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakouts with 1-week EMA50 trend filter and volume spike confirmation produce fewer, higher-quality trades that work in both bull and bear markets. The 12h TF reduces trade frequency to avoid fee drag, while the weekly EMA50 provides robust trend filtering. Volume spike ensures institutional participation. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data for Camarilla levels (R3/S3 from previous daily bar)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = close + 1.5*(high-low), S3 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_R3 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_S3 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (completed 1d bars only)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # 12h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
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
        
        # Camarilla R3/S3 breakout conditions
        breakout_up = close[i] > R3_aligned[i]   # Price breaks above R3
        breakout_down = close[i] < S3_aligned[i]  # Price breaks below S3
        
        # 1w EMA50 trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if breakout_up and uptrend and volume_spike:
            # Long signal: break above R3 + uptrend + volume spike
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif breakout_down and downtrend and volume_spike:
            # Short signal: break below S3 + downtrend + volume spike
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

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0