#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v4
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation on 12h timeframe.
Only long when price breaks above R1 and close > 1d EMA34, short when price breaks below S1 and close < 1d EMA34.
Volume confirmation requires volume > 1.5 * 20-period EMA volume.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
Designed to work in both bull and bear markets by combining price structure (Camarilla) with trend (1d EMA) and volume filters.
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
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].iloc[-2]  # Previous day close
    prev_high = df_1d['high'].iloc[-2]    # Previous day high
    prev_low = df_1d['low'].iloc[-2]      # Previous day low
    
    # Camarilla levels (R1, S1)
    range_val = prev_high - prev_low
    r1 = prev_close + range_val * 1.1 / 12
    s1 = prev_close - range_val * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (they change only at 1d boundaries)
    # Create arrays of R1 and S1 for each 1d bar, then align
    r1_array = np.full(len(df_1d), r1)
    s1_array = np.full(len(df_1d), s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_array)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_array)
    
    # Load 1d data for EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 periods for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above R1 + price > 1d EMA34 (trend up) + volume spike
        if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + price < 1d EMA34 (trend down) + volume spike
        elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to midpoint OR loss of volume confirmation
        elif position == 1 and (close[i] < (r1_aligned[i] + s1_aligned[i]) / 2 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > (r1_aligned[i] + s1_aligned[i]) / 2 or not volume_spike[i]):
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v4"
timeframe = "12h"
leverage = 1.0