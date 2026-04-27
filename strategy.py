#!/usr/bin/env python3
"""
6h_WeeklyPivot_TrendContinuation_v2
Hypothesis: Weekly pivot levels (calculated from prior week) act as dynamic support/resistance.
In trending markets (price > weekly pivot + weekly range), price tends to continue in direction
of breakout with momentum. Uses 1d trend filter and volume confirmation to avoid whipsaws.
Designed for low trade frequency (target 15-30/year) to minimize fee drag.
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
    
    # Calculate weekly pivot from prior week data (requires 1d data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close from prior week (using last 5 trading days)
    # Need at least 5 days of data
    if len(high_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's OHLC
    # We'll use rolling window of 5 days for weekly high/low/close
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1)    # Prior week
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1) # Prior week
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly range = H - L
    weekly_range = weekly_high - weekly_low
    
    # Support/resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # 1d trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all weekly indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need weekly data (5 days) + EMA50 (50) + volume avg (20)
    start_idx = max(50, 20)  # Weekly data handled by shift in calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot = weekly_pivot_aligned[i]
        r3 = weekly_r3_aligned[i]
        s3 = weekly_s3_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs EMA50 (1d)
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            # Long: break above weekly R3 with volume in uptrend
            if uptrend and vol_conf and close_val > r3:
                signals[i] = size
                position = 1
            # Short: break below weekly S3 with volume in downtrend
            elif downtrend and vol_conf and close_val < s3:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: price crosses below weekly pivot or trend reversal
            if close_val < pivot:  # Cross below pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price crosses above weekly pivot or trend reversal
            if close_val > pivot:  # Cross above pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_TrendContinuation_v2"
timeframe = "6h"
leverage = 1.0