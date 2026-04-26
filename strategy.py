#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: 4-hour Camarilla R3/S3 breakout with 12-hour EMA50 trend filter and volume spike confirmation.
Enters long when price breaks above R3 with bullish 12h trend and volume spike.
Enters short when price breaks below S3 with bearish 12h trend and volume spike.
Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Designed for 75-200 total trades over 4 years.
Works in both bull and bear markets by following the 12h trend direction only.
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
    
    # Calculate Camarilla pivot levels (R3, S3) on 12h timeframe
    # Use prior completed 12h bar to avoid look-ahead
    df_12h = get_htf_data(prices, '12h')
    
    # Prior 12h bar's OHLC for Camarilla calculation (shifted by 1)
    prior_high = np.roll(df_12h['high'].values, 1)
    prior_low = np.roll(df_12h['low'].values, 1)
    prior_close = np.roll(df_12h['close'].values, 1)
    prior_open = np.roll(df_12h['open'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    prior_open[0] = np.nan
    
    # Calculate pivot point and Camarilla levels
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_hl = prior_high - prior_low
    r3 = pivot + range_hl * 1.1 / 4
    s3 = pivot - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Load 12h data for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need 12h shift + 50-period EMA)
    start_idx = 1 + 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish 12h trend + volume spike
        if close[i] > r3_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish 12h trend + volume spike
        elif close[i] < s3_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
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

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0