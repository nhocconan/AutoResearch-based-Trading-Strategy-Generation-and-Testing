#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2x median) for entry.
Goes long when price breaks above R3 with volume spike and 1d trend bullish (price > EMA34).
Goes short when price breaks below S3 with volume spike and 1d trend bearish (price < EMA34).
Uses discrete position sizing (0.25) to minimize churn. Designed for 12-37 trades/year over 4 years.
Works in both bull and bear markets by following 1d trend filter.
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
    
    # Calculate Camarilla levels for 12h (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous 12h bar's high, low, close
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_12h_prev = np.roll(h_12h, 1)
    l_12h_prev = np.roll(l_12h, 1)
    c_12h_prev = np.roll(c_12h, 1)
    # First bar will have rolled values from last - set to nan so they don't trigger
    h_12h_prev[0] = np.nan
    l_12h_prev[0] = np.nan
    c_12h_prev[0] = np.nan
    
    # Calculate Camarilla R3 and S3 levels
    rng_12h = h_12h_prev - l_12h_prev
    r3_12h = c_12h_prev + (rng_12h * 1.1 / 4)
    s3_12h = c_12h_prev - (rng_12h * 1.1 / 4)
    
    # Align to 12h primary timeframe (no additional delay needed as levels are based on completed bar)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Volume spike: volume > 2x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median and EMA)
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_34_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R3 + volume spike + bullish 1d trend
        if close[i] > r3_12h_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S3 + volume spike + bearish 1d trend
        elif close[i] < s3_12h_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to median levels)
        elif position == 1 and close[i] < (r3_12h_aligned[i] + s3_12h_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (r3_12h_aligned[i] + s3_12h_aligned[i]) / 2:
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

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0