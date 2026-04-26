#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Camarilla pivot levels from 1d timeframe, with breakout continuation at R4/S4 levels.
Only enter long when price breaks above R4 with volume confirmation and weekly trend up (price > weekly EMA50).
Enter short when price breaks below S4 with volume confirmation and weekly trend down (price < weekly EMA50).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
Works in bull markets via breakout continuation and in bear markets via breakdown continuation.
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
    
    # Load 1d data for Camarilla pivots and weekly trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 1d bar (HLC of completed daily bar)
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+CLOSE)/3 of previous completed daily bar
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # Typical price of previous completed daily bar
    typical_price = (df_1d_high + df_1d_low + df_1d_close) / 3.0
    # Range of previous completed daily bar
    daily_range = df_1d_high - df_1d_low
    # Camarilla R4 and S4 levels
    r4 = typical_price + (daily_range * 1.1 / 2.0)
    s4 = typical_price - (daily_range * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly trend filter: EMA50 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
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
        
        # Long logic: price breaks above R4 + volume spike + weekly trend up
        if close[i] > r4_aligned[i] and volume_spike[i] and close[i] > ema_50_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S4 + volume spike + weekly trend down
        elif close[i] < s4_aligned[i] and volume_spike[i] and close[i] < ema_50_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to Camarilla H3/L3 levels or loss of volume
        elif position == 1 and (close[i] < r4_aligned[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > s4_aligned[i] or not volume_spike[i]):
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

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0