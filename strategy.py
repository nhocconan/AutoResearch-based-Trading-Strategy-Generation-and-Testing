#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (weekly pivots from previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    # Get previous week's OHLC (shifted by 1 to avoid look-ahead)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Calculate pivot points for previous week
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_ltf_to_hlf(prices, df_1w, pivot)
    r1_aligned = align_ltf_to_hlf(prices, df_1w, r1)
    s1_aligned = align_ltf_to_hlf(prices, df_1w, s1)
    r2_aligned = align_ltf_to_hlf(prices, df_1w, r2)
    s2_aligned = align_ltf_to_hlf(prices, df_1w, s2)
    r3_aligned = align_ltf_to_hlf(prices, df_1w, r3)
    s3_aligned = align_ltf_to_hlf(prices, df_1w, s3)
    
    # Get daily trend filter (1d EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_ltf_to_hlf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above R1 AND above daily EMA50 + volume
            if (close[i] > r1_aligned[i] and close[i] > ema_50_1d_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 AND below daily EMA50 + volume
            elif (close[i] < s1_aligned[i] and close[i] < ema_50_1d_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: price falls below pivot OR below daily EMA50
                if close[i] < pivot_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price rises above pivot OR above daily EMA50
                if close[i] > pivot_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

# Note: The align_ltf_to_hlf function doesn't exist in mtf_data.
# Let me correct this to use the proper function name align_htf_to_ltf.