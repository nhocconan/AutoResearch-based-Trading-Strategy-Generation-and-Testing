#!/usr/bin/env python3
name = "6h_WeeklyPivot_VolumeBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly pivot points (calculated from previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high, low, close
    prev_week_high = df_1w['high'].iloc[-2]
    prev_week_low = df_1w['low'].iloc[-2]
    prev_week_close = df_1w['close'].iloc[-2]
    
    # Pivot point calculation
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot)
    
    # Broadcast pivot levels to all bars
    pivot_arr = np.full(n, pivot)
    r1_arr = np.full(n, r1)
    s1_arr = np.full(n, s1)
    r2_arr = np.full(n, r2)
    s2_arr = np.full(n, s2)
    r3_arr = np.full(n, r3)
    s3_arr = np.full(n, s3)
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detector: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for daily EMA50
    
    for i in range(start_idx, n):
        # Skip if daily trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume spike + price above daily EMA50
            if (close[i] > r1_arr[i] and 
                vol_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume spike + price below daily EMA50
            elif (close[i] < s1_arr[i] and 
                  vol_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below pivot or loss of daily uptrend
            if (close[i] < pivot_arr[i] or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above pivot or loss of daily downtrend
            if (close[i] > pivot_arr[i] or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals