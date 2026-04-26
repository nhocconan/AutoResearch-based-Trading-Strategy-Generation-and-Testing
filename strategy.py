#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Daily Camarilla R3/S3 breakouts with weekly EMA34 trend filter and volume spike capture major swing continuations. Uses 1d timeframe for lower frequency (7-25 trades/year) to minimize fee drag. Weekly trend ensures alignment with higher timeframe momentum. Works in both bull (breakouts with trend) and bear (failed breaks reverse) markets.
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
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load daily data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Need at least 2 days for previous day calculation
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1 / 4)
    s3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection on 1d (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter (EMA34)
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Long logic: price breaks above R3 with volume spike + in uptrend
        if close[i] > r3_aligned[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S3 with volume spike + in downtrend
        elif close[i] < s3_aligned[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite level or trend weakens
        elif position == 1 and (close[i] < s3_aligned[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r3_aligned[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0