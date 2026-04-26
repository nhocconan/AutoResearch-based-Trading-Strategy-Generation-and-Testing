#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: Daily Camarilla R3/S3 breakout with weekly trend filter and volume confirmation.
Enters long when price breaks above R3 with bullish weekly trend and volume spike.
Enters short when price breaks below S3 with bearish weekly trend and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 30-100 trades over 4 years.
Works in both bull and bear markets by following the weekly trend direction only.
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
    
    # Calculate prior day's Camarilla levels (using prior daily bar)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's OHLC (shifted by 1 to use completed daily bar)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Camarilla calculations
    rang = prior_high - prior_low
    camarilla_r3 = prior_close + rang * 1.1 / 4
    camarilla_s3 = prior_close - rang * 1.1 / 4
    camarilla_r4 = prior_close + rang * 1.1 / 2
    camarilla_s4 = prior_close - rang * 1.1 / 2
    
    # Align Camarilla levels to daily timeframe (no shift needed as we use prior day)
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need prior daily bar + weekly EMA50)
    start_idx = 50 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish weekly trend + volume spike
        if close[i] > r3_1d[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish weekly trend + volume spike
        elif close[i] < s3_1d[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s3_1d[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r3_1d[i]:
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

name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0