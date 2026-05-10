#!/usr/bin/env python3
# 6H_Weekly_Pivot_Range_Breakout
# Hypothesis: Trade breakouts of weekly pivot ranges on 6h timeframe with volume confirmation.
# Long when price breaks above weekly R1 with volume > 1.5x average.
# Short when price breaks below weekly S1 with volume > 1.5x average.
# Uses 1d trend filter: only trade long in 1d uptrend, short in 1d downtrend.
# Weekly pivots calculated from prior week's OHLC. Works in bull/bear by following 1d trend.
# Target: 15-30 trades/year per symbol.

name = "6H_Weekly_Pivot_Range_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Volume average (24-period for 6h = 4 days)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=24, min_periods=24).mean().values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Standard formula: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use prior week's data (avoid look-ahead)
    weekly_high_shift = np.roll(weekly_high, 1)
    weekly_low_shift = np.roll(weekly_low, 1)
    weekly_close_shift = np.roll(weekly_close, 1)
    # First value remains 0 (no prior week data)
    weekly_high_shift[0] = 0
    weekly_low_shift[0] = 0
    weekly_close_shift[0] = 0
    
    # Calculate pivots
    pivot = (weekly_high_shift + weekly_low_shift + weekly_close_shift) / 3.0
    r1 = 2 * pivot - weekly_low_shift
    s1 = 2 * pivot - weekly_high_shift
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(daily_uptrend_aligned[i]) or 
            np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: 1d uptrend + price breaks above weekly R1 + volume
            if daily_up and close[i] > r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d downtrend + price breaks below weekly S1 + volume
            elif daily_down and close[i] < s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend changes or price returns to pivot
            if not daily_up or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend changes or price returns to pivot
            if not daily_down or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals