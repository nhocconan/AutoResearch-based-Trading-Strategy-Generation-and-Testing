#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation on 12h timeframe.
Enters long when price breaks above R3 with bullish weekly trend and volume spike.
Enters short when price breaks below S3 with bearish weekly trend and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
Designed to work in both bull and bear markets by following the 1w trend direction only.
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
    
    # Calculate prior week's Camarilla levels (using prior completed weekly bar)
    df_1w = get_htf_data(prices, '1w')
    
    # Prior week's OHLC (shifted by 1 to use completed weekly bar)
    prior_close = np.roll(df_1w['close'].values, 1)
    prior_high = np.roll(df_1w['high'].values, 1)
    prior_low = np.roll(df_1w['low'].values, 1)
    prior_close[0] = np.nan  # First bar has no prior
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Camarilla calculations
    rang = prior_high - prior_low
    camarilla_r3 = prior_close + rang * 1.1 / 4
    camarilla_s3 = prior_close - rang * 1.1 / 4
    camarilla_r4 = prior_close + rang * 1.1 / 2
    camarilla_s4 = prior_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    r4_12h = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s4_12h = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Load 1w data for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need prior weekly bar + EMA34)
    start_idx = 34 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish 1w trend + volume spike
        if close[i] > r3_12h[i] and close[i] > ema_34_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish 1w trend + volume spike
        elif close[i] < s3_12h[i] and close[i] < ema_34_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s3_12h[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r3_12h[i]:
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

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0