#!/usr/bin/env python3
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
    
    # Get 1d data for weekly pivot calculation (using daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using last completed week (Monday open to Friday close)
    # We'll use 5-day rolling window to approximate weekly OHLC
    def calculate_weekly_pivot(high_arr, low_arr, close_arr):
        if len(high_arr) < 5:
            return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
        
        # Use last 5 days for weekly OHLC (approximation)
        weekly_high = np.max(high_arr[-5:])
        weekly_low = np.min(low_arr[-5:])
        weekly_close = close_arr[-1]  # Most recent close
        
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - pivot)
        
        return np.array([pivot]), np.array([r1]), np.array([s1]), np.array([r2]), np.array([s2]), np.array([r3]), np.array([s3])
    
    # Get weekly pivot values (updated weekly)
    pivot_val, r1_val, s1_val, r2_val, s2_val, r3_val, s3_val = calculate_weekly_pivot(high_1d, low_1d, close_1d)
    
    # If we don't have enough data for weekly pivot, use daily pivot as fallback
    if np.isnan(pivot_val[0]):
        # Daily pivot calculation
        daily_high = high_1d[-1]
        daily_low = low_1d[-1]
        daily_close = close_1d[-1]
        
        pivot_val = np.array([(daily_high + daily_low + daily_close) / 3.0])
        r1_val = np.array([2 * pivot_val[0] - daily_low])
        s1_val = np.array([2 * pivot_val[0] - daily_high])
        r2_val = np.array([pivot_val[0] + (daily_high - daily_low)])
        s2_val = np.array([pivot_val[0] - (daily_high - daily_low)])
        r3_val = np.array([daily_high + 2 * (pivot_val[0] - daily_low)])
        s3_val = np.array([daily_low - 2 * (daily_high - pivot_val[0])])
    
    # Get 6h data for EMA20 trend filter (more responsive for 6h timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA20
    ema_period = 20
    ema_6h = np.full(len(close_6h), np.nan)
    if len(close_6h) >= ema_period:
        ema_6h[ema_period - 1] = np.mean(close_6h[:ema_period])
        for i in range(ema_period, len(close_6h)):
            ema_6h[i] = (close_6h[i] * (2 / (ema_period + 1)) + 
                        ema_6h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 6h timeframe (our primary timeframe)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), pivot_val[0]))
    r1_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r1_val[0]))
    s1_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s1_val[0]))
    r2_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r2_val[0]))
    s2_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s2_val[0]))
    r3_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r3_val[0]))
    s3_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s3_val[0]))
    ema_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_6h)
    
    # Volume filter: current volume > 1.5x 20-period average (more reasonable threshold)
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA and volume MA
    start_idx = max(ema_period, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_6h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        ema = ema_6h_aligned[i]
        pivot = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume, above EMA20 (bullish bias)
            if (price > r1 and 
                vol_ratio > 1.5 and 
                price > ema):
                signals[i] = size
                position = 1
            # Short: Price breaks below S1 with volume, below EMA20 (bearish bias)
            elif (price < s1 and 
                  vol_ratio > 1.5 and 
                  price < ema):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below S1 (mean reversion) or EMA20 turns bearish
            if (price < s1 or 
                price < ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price breaks above R1 (mean reversion) or EMA20 turns bullish
            if (price > r1 or 
                price > ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_PivotPoints_R1S1_Breakout_EMA20_Volume"
timeframe = "6h"
leverage = 1.0