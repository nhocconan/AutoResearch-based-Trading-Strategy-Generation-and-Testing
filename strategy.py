#!/usr/bin/env python3
"""
1d_WeeklyPivot_TrendBreakout
Hypothesis: Trade breakouts of weekly pivot levels (R1/S1) with daily trend filter and volume confirmation. Weekly pivots provide strong support/resistance levels that often hold in both bull and bear markets. Breakouts above/below these levels with volume and daily trend alignment capture momentum moves. Designed for low trade frequency (<20/year) to minimize fee drag while capturing significant moves.
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # Handle first bar
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    # Weekly pivot levels
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1_1w = 2.0 * pivot_1w - prev_low_1w
    s1_1w = 2.0 * pivot_1w - prev_high_1w
    
    # Daily EMA trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_period = 50
    if len(close_1d) >= ema_period:
        ema_1d = np.zeros_like(close_1d)
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    else:
        ema_1d = np.full_like(close_1d, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    vol_period = 20
    for i in range(vol_period, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align weekly pivot levels and daily EMA to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above daily EMA
            if close[i] > r1_1w_aligned[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below daily EMA
            elif close[i] < s1_1w_aligned[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (opposite pivot) or below daily EMA
            if close[i] < s1_1w_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 (opposite pivot) or above daily EMA
            if close[i] > r1_1w_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_TrendBreakout"
timeframe = "1d"
leverage = 1.0