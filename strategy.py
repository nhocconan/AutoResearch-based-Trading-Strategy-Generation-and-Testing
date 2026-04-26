#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v1
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe with 1d EMA34 trend filter and volume confirmation.
Goes long when price breaks above R1 with volume spike and price > 1d EMA34.
Goes short when price breaks below S1 with volume spike and price < 1d EMA34.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by combining structure (Camarilla pivots) with trend (1d EMA) and volume filters.
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
    
    # Calculate Camarilla pivot levels from daily timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on previous day's range
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need daily data for Camarilla and EMA)
    start_idx = max(34, 20) + 1
    
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
        
        # Long logic: price breaks above R1 + volume spike + price > 1d EMA34 (uptrend)
        if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume spike + price < 1d EMA34 (downtrend)
        elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to mean (close to pivot) or loss of volume confirmation
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0