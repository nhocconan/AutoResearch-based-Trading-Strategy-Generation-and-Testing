#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot + 1d Trend + Volume Spike
# Uses weekly pivot levels (R4/S4) as major support/resistance for breakouts.
# Filters with daily EMA50 trend and volume spike to avoid false breakouts.
# Designed for low frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.

name = "6h_WeeklyPivot_R4S4_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels (R4, S4)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (R4, S4)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Previous week's values for current week's pivot
    prev_high_w = np.roll(high_w, 1)
    prev_low_w = np.roll(low_w, 1)
    prev_close_w = np.roll(close_w, 1)
    prev_high_w[0] = np.nan
    prev_low_w[0] = np.nan
    prev_close_w[0] = np.nan
    
    # Weekly pivot point
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    # Weekly R4 and S4 (strong breakout levels)
    R4_w = prev_high_w + 3 * (pivot_w - prev_low_w)
    S4_w = prev_low_w - 3 * (prev_high_w - pivot_w)
    
    # Align weekly pivot levels to 6h timeframe
    R4_w_aligned = align_htf_to_ltf(prices, df_w, R4_w)
    S4_w_aligned = align_htf_to_ltf(prices, df_w, S4_w)
    
    # Get daily data for EMA50 trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    close_d = df_d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Volume spike (2.0x 24-period EMA ≈ 4 days of 6h bars)
    vol_ma = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R4_w_aligned[i]) or np.isnan(S4_w_aligned[i]) or 
            np.isnan(ema50_d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly R4 with daily uptrend and volume spike
            if (close[i] > R4_w_aligned[i] and 
                close[i] > ema50_d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S4 with daily downtrend and volume spike
            elif (close[i] < S4_w_aligned[i] and 
                  close[i] < ema50_d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S4 or trend fails
            if (close[i] < S4_w_aligned[i] or 
                close[i] < ema50_d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R4 or trend fails
            if (close[i] > R4_w_aligned[i] or 
                close[i] > ema50_d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals