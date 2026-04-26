#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Confluence
Hypothesis: 12h Camarilla R1/S1 breakout with 1-week trend filter and volume confirmation.
Uses weekly Camarilla-like structure derived from 1w high-low range to identify key levels,
combined with 1w EMA50 trend filter to ensure trades align with higher timeframe momentum.
Volume spike confirms participation. Designed for 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by following 1w trend while capturing mean-reversion
at extreme 12h Camarilla levels. Discrete position sizing (0.0, ±0.25) minimizes fee drag.
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w-based Camarilla levels (using 1w range scaled to 12h)
    # Camarilla: (H-L)*1.1/12 gives key levels; we use 1w range for structure
    hl_range_1w = (df_1w['high'].values - df_1w['low'].values)
    camarilla_width = hl_range_1w * 1.1 / 12
    camarilla_R1_1w = df_1w['close'].values + camarilla_width * 1
    camarilla_S1_1w = df_1w['close'].values - camarilla_width * 1
    
    # Align 1w Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R1_1w)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S1_1w)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume (on 12h data)
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup: max of 1w EMA50 (50) and volume EMA (20)
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Close breaks above Camarilla R1 + price > 1w EMA50 (uptrend) + volume spike
        if close[i] > camarilla_R1_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below Camarilla S1 + price < 1w EMA50 (downtrend) + volume spike
        elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses 1w EMA50 in opposite direction
        elif position == 1 and close[i] < ema_50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_50_1w_aligned[i]:
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

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Confluence"
timeframe = "12h"
leverage = 1.0