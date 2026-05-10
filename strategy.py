#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Use weekly Camarilla R1/S1 levels for breakout entries in the direction of weekly EMA trend, filtered by volume spike.
Designed for 12h timeframe to target ~15-30 trades/year, avoiding excessive fee drift.
Weekly timeframe provides strong trend filter that works in both bull and bear markets.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (using prior week)
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Camarilla equations
    range_prev = high_prev - low_prev
    camarilla_r1 = close_prev + range_prev * 1.1 / 12
    camarilla_s1 = close_prev - range_prev * 1.1 / 12
    
    # Align weekly Camarilla levels to 12h timeframe (wait for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Weekly EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Get 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly Camarilla (needs 1 week), EMA34 (34 bars), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA34 (uptrend) AND price breaks above Camarilla R1 with volume spike
            if close[i] > ema_34_aligned[i] and high[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA34 (downtrend) AND price breaks below Camarilla S1 with volume spike
            elif close[i] < ema_34_aligned[i] and low[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla pivot (central level) OR trend turns bearish
            # Calculate Camarilla pivot for exit: (H+L+2*C)/4
            camarilla_pivot = (high_prev + low_prev + 2 * close_prev) / 4
            pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
            if low[i] < pivot_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla pivot OR trend turns bullish
            camarilla_pivot = (high_prev + low_prev + 2 * close_prev) / 4
            pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
            if high[i] > pivot_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals