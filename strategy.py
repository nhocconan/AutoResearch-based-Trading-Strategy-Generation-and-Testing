#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Volume Spike + 6h EMA50 Trend
Hypothesis: Weekly pivot points act as strong support/resistance. Breakouts above weekly R1 or below S1 with daily volume confirmation and aligned 6h EMA50 trend capture institutional flow. Works in bull/bear by trading with the 6h EMA50 trend filter. Targets 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily data for volume spike confirmation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day volume MA for volume spike
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 6h EMA50 for trend filter (call ONCE before loop)
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, volume MA, and weekly data
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        ema_50_val = ema_50[i]
        
        # Trend filter: price relative to 6h EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.5 * 20-day average
        volume_confirm = curr_volume > 2.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above weekly R1 with volume confirmation in uptrend
            long_breakout = (curr_close > r1_val) and volume_confirm and uptrend
            # Short: price breaks below weekly S1 with volume confirmation in downtrend
            short_breakout = (curr_close < s1_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below weekly pivot OR EMA50 trend turns down
            if curr_close < pivot_val or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly pivot OR EMA50 trend turns up
            if curr_close > pivot_val or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_1dVolumeSpike_EMA50_Trend"
timeframe = "6h"
leverage = 1.0