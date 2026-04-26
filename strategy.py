#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike (>2x median) for entry.
Goes long when price breaks above R3 with volume spike and 12h trend bullish (price > EMA50).
Goes short when price breaks below S3 with volume spike and 12h trend bearish (price < EMA50).
Uses discrete position sizing (0.25) to minimize churn. Designed for 75-200 total trades over 4 years.
Works in both bull and bear markets by following 12h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 4h (based on previous 4h bar)
    # We need to calculate these on 4h data then align to 4h primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous 4h bar's high, low, close
    # Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, etc.
    # But for breakout we use R3 and S3: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_4h_prev = np.roll(h_4h, 1)
    l_4h_prev = np.roll(l_4h, 1)
    c_4h_prev = np.roll(c_4h, 1)
    # First bar will have rolled values from last - set to nan so they don't trigger
    h_4h_prev[0] = np.nan
    l_4h_prev[0] = np.nan
    c_4h_prev[0] = np.nan
    
    # Calculate Camarilla R3 and S3 levels
    rng_4h = h_4h_prev - l_4h_prev
    r3_4h = c_4h_prev + (rng_4h * 1.1 / 4)
    s3_4h = c_4h_prev - (rng_4h * 1.1 / 4)
    
    # Align to 4h primary timeframe (no additional delay needed as levels are based on completed bar)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume spike: volume > 2x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    # Load 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median and EMA)
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_12h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R3 + volume spike + bullish 12h trend
        if close[i] > r3_4h_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S3 + volume spike + bearish 12h trend
        elif close[i] < s3_4h_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to median levels)
        elif position == 1 and close[i] < (r3_4h_aligned[i] + s3_4h_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (r3_4h_aligned[i] + s3_4h_aligned[i]) / 2:
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

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0