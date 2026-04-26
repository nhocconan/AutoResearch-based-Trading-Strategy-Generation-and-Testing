#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Trade 12h Camarilla R1/S1 breakouts in direction of 1w EMA50 trend with volume spike filter.
Uses weekly EMA50 for primary trend alignment (more stable than daily in bear markets),
Camarilla pivot levels from daily timeframe for precise entry/exit levels,
and volume spike on 12h for breakout conviction.
Designed for lower trade frequency (target 12-37/year) to minimize fee drag on 12h timeframe.
Works in both bull and bear markets via strong trend filter + volume confirmation.
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
    
    # Get 1d data for Camarilla pivots (daily structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend (weekly trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_r1 = camarilla_pivot + (range_1d * 1.0 / 12)
    camarilla_s1 = camarilla_pivot - (range_1d * 1.0 / 12)
    camarilla_r2 = camarilla_pivot + (range_1d * 2.0 / 12)
    camarilla_s2 = camarilla_pivot - (range_1d * 2.0 / 12)
    camarilla_r3 = camarilla_pivot + (range_1d * 3.0 / 12)
    camarilla_s3 = camarilla_pivot - (range_1d * 3.0 / 12)
    camarilla_r4 = camarilla_pivot + (range_1d * 4.0 / 12)
    camarilla_s4 = camarilla_pivot - (range_1d * 4.0 / 12)
    
    # Align Camarilla levels to 12h timeframe (prior 1d's levels available at 1d close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 2.0x 20-period average on 12h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1w EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + price above 1w EMA50 + volume spike
            long_breakout = close[i] > camarilla_r1_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i]
            
            # Short: price breaks below Camarilla S1 + price below 1w EMA50 + volume spike
            short_breakout = close[i] < camarilla_s1_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Camarilla S1 OR trend turns bearish (price below EMA)
            if (close[i] < camarilla_s1_aligned[i] or not price_above_ema):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 OR trend turns bullish (price above EMA)
            if (close[i] > camarilla_r1_aligned[i] or not price_below_ema):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0