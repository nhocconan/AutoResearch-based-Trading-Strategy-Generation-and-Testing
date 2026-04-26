#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_1dTrend_VolumeSpike
Hypothesis: On 6h timeframe, enter long when price reverses up from weekly S1 pivot AND 1d trend is up (close > EMA34) AND volume > 2.0x 20-period average volume. Enter short when price reverses down from weekly R1 pivot AND 1d trend is down (close < EMA34) AND volume > 2.0x 20-period average volume. Exit on 1d trend reversal or price reaching opposite weekly pivot (R1 for longs, S1 for shorts). Weekly pivots calculated from prior week's OHLC. This strategy targets weekly mean reversion within the 1d trend, which works in both bull and bear markets by fading weekly extremes while respecting the daily trend filter. Volume spike confirms institutional interest at pivot levels. Target: 12-30 trades/year (50-120 over 4 years).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous weekly values
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Weekly pivot calculation
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    
    # Align weekly pivots and 1d EMA to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Reversal conditions: price touching weekly pivot and reversing
        # Long: price touches or goes below S1 then reverses up (close > S1)
        touch_s1 = low[i] <= s1_aligned[i]
        reverse_up = close[i] > s1_aligned[i]  # Closed back above S1
        
        # Short: price touches or goes above R1 then reverses down (close < R1)
        touch_r1 = high[i] >= r1_aligned[i]
        reverse_down = close[i] < r1_aligned[i]  # Closed back below R1
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: reversal up from S1 + volume spike + 1d uptrend
            long_signal = touch_s1 and reverse_up and volume_spike[i] and trend_uptrend
            
            # Short: reversal down from R1 + volume spike + 1d downtrend
            short_signal = touch_r1 and reverse_down and volume_spike[i] and trend_downtrend
            
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
            # Exit: 1d trend change to downtrend OR price reaching weekly R1 (target)
            if not trend_uptrend or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 1d trend change to uptrend OR price reaching weekly S1 (target)
            if not trend_downtrend or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Weekly_Pivot_Reversal_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0