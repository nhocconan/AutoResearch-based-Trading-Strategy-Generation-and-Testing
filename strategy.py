#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume confirmation (>2.0x EMA20 volume).
Enters long when price breaks above R1 with bullish 4h trend and volume spike.
Enters short when price breaks below S1 with bearish 4h trend and volume spike.
Exits when price reverts to opposite Camarilla level (S1 for longs, R1 for shorts).
Designed for 60-150 total trades over 4 years (15-37/year) to avoid fee drag on 1h timeframe.
Uses discrete position sizing (0.20) to minimize churn. Works in both bull and bear markets by following 4h trend.
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
    
    # Calculate Camarilla pivot levels (R1, S1) on 1h timeframe using prior completed 1h bar
    # Use shift(1) on 1h data to avoid look-ahead
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Calculate pivot point and Camarilla levels (R1, S1)
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_hl = prior_high - prior_low
    r1 = pivot + range_hl * 1.1 / 4.0
    s1 = pivot - range_hl * 1.1 / 4.0
    
    # Load 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for volume confirmation (>2.0x EMA20 volume)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 1h shift + 50-period EMA)
    start_idx = max(1 + 50, 1 + 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 + bullish 4h trend + volume spike
        if close[i] > r1[i] and close[i] > ema_50_4h_aligned[i] and volume_spike_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S1 + bearish 4h trend + volume spike
        elif close[i] < s1[i] and close[i] < ema_50_4h_aligned[i] and volume_spike_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s1[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1[i]:
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0