#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Weekly Camarilla R3/S3 breakouts with 1w EMA34 trend filter and volume spike confirmation on 6h timeframe produce high-quality trades with low frequency (target: 80-120 total over 4 years). Works in both bull and bear markets via trend filter. Uses actual weekly data from Binance parquet via mtf_data helper.
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
    
    # Load 1w data ONCE before loop for HTF trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    camarilla_R3 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_S3 = close_1w - 1.1 * (high_1w - low_1w) / 2
    camarilla_R4 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_S4 = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align Camarilla levels to 6h timeframe (completed weekly bars only)
    R3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S4)
    
    # 6h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
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
        
        # 1w EMA34 trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Additional filter: avoid false breakouts by requiring price to stay beyond R4/S4 for confirmation
        # This reduces whipsaws in ranging markets
        strong_breakout_up = breakout_up and close[i] > R4_aligned[i] * 0.999  # Allow small buffer
        strong_breakout_down = breakout_down and close[i] < S4_aligned[i] * 1.001  # Allow small buffer
        
        if strong_breakout_up and uptrend and volume_spike:
            # Long signal: strong break above R3 + uptrend + volume spike
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif strong_breakout_down and downtrend and volume_spike:
            # Short signal: strong break below S3 + downtrend + volume spike
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

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0