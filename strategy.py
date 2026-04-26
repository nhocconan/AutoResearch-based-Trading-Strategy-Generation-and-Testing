#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Bounce_1dTrend_VolumeFilter
Hypothesis: Price bounces off Camarilla pivot levels (R3/S3) in alignment with 1d trend and volume confirmation.
Enters long when price touches S3 with bullish 1d trend and above-average volume.
Enters short when price touches R3 with bearish 1d trend and above-average volume.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 75-150 total trades over 4 years.
Works in both bull and bear markets by following the 1d trend direction only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's OHLC (use completed daily bar, shifted by 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Camarilla R3 and S3 levels from prior day
    rang = prior_high - prior_low
    camarilla_r3 = prior_close + rang * 1.1 / 4
    camarilla_s3 = prior_close - rang * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need prior daily bar + EMA34)
    start_idx = 34 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: touch or penetrate S3 + bullish 1d trend + volume filter
        if low[i] <= s3_4h[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: touch or penetrate R3 + bearish 1d trend + volume filter
        elif high[i] >= r3_4h[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level (mean reversion)
        elif position == 1 and close[i] >= (r3_4h[i] + s3_4h[i]) / 2:  # midpoint
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] <= (r3_4h[i] + s3_4h[i]) / 2:  # midpoint
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

name = "4h_Camarilla_Pivot_Bounce_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0