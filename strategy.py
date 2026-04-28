# 6h_WeeklyPivot_DailyTrend_Filter
# Hypothesis: Weekly pivots provide strong institutional support/resistance levels. 
# In bull markets, price bounces from weekly support; in bear markets, price rejects at weekly resistance.
# Daily trend filters ensure we only trade in the direction of higher timeframe momentum.
# Weekly pivots are calculated from prior week's OHLC and are static for the week, eliminating look-ahead.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag.
# Uses 6h timeframe for balance of signal quality and trade frequency.

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
    
    # Get weekly data once for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_daily['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Align daily EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to weekly pivot levels
        price = close[i]
        at_pivot = abs(price - pivot_aligned[i]) < 0.001 * pivot_aligned[i]  # Within 0.1% of pivot
        near_s1 = price > s1_aligned[i] and price < s1_aligned[i] * 1.002  # Within 0.2% above S1
        near_s2 = price > s2_aligned[i] and price < s2_aligned[i] * 1.002  # Within 0.2% above S2
        near_s3 = price > s3_aligned[i] and price < s3_aligned[i] * 1.002  # Within 0.2% above S3
        near_r1 = price < r1_aligned[i] and price > r1_aligned[i] * 0.998  # Within 0.2% below R1
        near_r2 = price < r2_aligned[i] and price > r2_aligned[i] * 0.998  # Within 0.2% below R2
        near_r3 = price < r3_aligned[i] and price > r3_aligned[i] * 0.998  # Within 0.2% below R3
        
        # Trend filter: daily EMA50
        uptrend = price > ema_50_aligned[i]
        downtrend = price < ema_50_aligned[i]
        
        # Entry logic: 
        # Long: price near weekly support (S1-S3) in uptrend
        # Short: price near weekly resistance (R1-R3) in downtrend
        long_entry = (near_s1 or near_s2 or near_s3) and uptrend
        short_entry = (near_r1 or near_r2 or near_r3) and downtrend
        
        # Exit logic: opposite signal or price moves away from pivot area
        long_exit = not uptrend or (price > pivot_aligned[i] * 1.01)  # Exit if trend turns or price moves 1% above pivot
        short_exit = not downtrend or (price < pivot_aligned[i] * 0.99)  # Exit if trend turns or price moves 1% below pivot
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_DailyTrend_Filter"
timeframe = "6h"
leverage = 1.0