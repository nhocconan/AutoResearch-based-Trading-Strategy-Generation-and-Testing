#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout_Trend_Volume
Hypothesis: Daily timeframe strategy using weekly Camarilla pivot levels (R1/S1) for breakout signals,
filtered by weekly EMA50 trend direction and daily volume confirmation. Weekly HTF ensures alignment
with multi-week trend, reducing false breakouts in choppy markets. Designed for 30-100 total trades
over 4 years (7-25/year) with discrete position sizing (0.0, ±0.25) to minimize fee drag. Works in
both bull and bear markets by only taking trades in direction of weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels from previous weekly bar
    camarilla_range = (df_1w['high'].values - df_1w['low'].values) * 1.1 / 12
    camarilla_R1 = df_1w['close'].values + camarilla_range * 1
    camarilla_S1 = df_1w['close'].values - camarilla_range * 1
    
    # Align weekly Camarilla levels to daily timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S1)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume confirmation: volume > 1.8 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(50, 50, 20) + 1
    
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
        
        # Long logic: Close breaks above weekly Camarilla R1 + price > weekly EMA50 (uptrend) + volume spike
        if close[i] > camarilla_R1_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below weekly Camarilla S1 + price < weekly EMA50 (downtrend) + volume spike
        elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses weekly EMA50 in opposite direction
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

name = "1d_Weekly_Pivot_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0