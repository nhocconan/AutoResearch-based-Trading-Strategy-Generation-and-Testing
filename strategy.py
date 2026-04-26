#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 1h with 4h trend filter and volume spike confirmation.
Enters long when price breaks above R3 with bullish 4h trend (price > EMA20) and volume spike.
Enters short when price breaks below S3 with bearish 4h trend (price < EMA20) and volume spike.
Uses discrete position sizing (0.0, ±0.20) to minimize fee churn. Target: 60-150 total trades over 4 years.
Uses 4h for signal direction, 1h only for entry timing. Session filter 08-20 UTC to reduce noise.
Works in both bull and bear markets by following the 4h trend direction only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate prior day's Camarilla levels (using prior completed daily bar)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's OHLC (shifted by 1 to use completed daily bar)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close[0] = np.nan  # First bar has no prior
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Camarilla calculations
    rang = prior_high - prior_low
    camarilla_r3 = prior_close + rang * 1.1 / 4
    camarilla_s3 = prior_close - rang * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_1h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA20 trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need prior daily bar + EMA20)
    start_idx = 20 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish 4h trend + volume spike
        if close[i] > r3_1h[i] and close[i] > ema_20_4h_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish 4h trend + volume spike
        elif close[i] < s3_1h[i] and close[i] < ema_20_4h_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s3_1h[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r3_1h[i]:
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

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0