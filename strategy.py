#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above R3 with bullish daily trend and volume spike.
Enters short when price breaks below S3 with bearish daily trend and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 20-50 trades/year.
Works in both bull and bear markets by following the daily trend direction only.
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
    
    # Calculate Camarilla levels from prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prior_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else np.nan
    prior_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else np.nan
    prior_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else np.nan
    
    # Camarilla R3 and S3 levels
    camarilla_range = prior_high - prior_low
    r3 = prior_close + camarilla_range * 1.1 / 4
    s3 = prior_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (they update only when new daily bar completes)
    r3_array = np.full(len(df_1d), r3)
    s3_array = np.full(len(df_1d), s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_array)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_array)
    
    # Daily EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 34-period EMA + volume EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish daily trend + volume spike
        if close[i] > r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish daily trend + volume spike
        elif close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s3_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r3_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0