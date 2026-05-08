#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Trend_Volume_Breakout"
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
    
    # Get weekly data for pivot calculation (using weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (classic formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivots for each week
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily trend filter (using 6h data but calculating EMA on daily equivalent)
    # We'll use 4-period EMA on 6h data ≈ 1 day (4*6h = 24h)
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=4, adjust=False, min_periods=4).mean().values
    ema_slow = close_series.ewm(span=16, adjust=False, min_periods=16).mean().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R3 with upward trend and volume
            long_breakout = close[i] > r3_aligned[i]
            uptrend = ema_fast[i] > ema_slow[i]
            
            # Short breakdown: price breaks below weekly S3 with downward trend and volume
            short_breakdown = close[i] < s3_aligned[i]
            downtrend = ema_fast[i] < ema_slow[i]
            
            if long_breakout and uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakdown and downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly pivot or trend reverses
            if close[i] < pivot_aligned[i] or ema_fast[i] < ema_slow[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly pivot or trend reverses
            if close[i] > pivot_aligned[i] or ema_fast[i] > ema_slow[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals