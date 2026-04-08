#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend + Volume Confirmation
Hypothesis: Weekly pivot points (R4/S3) provide institutional support/resistance levels.
In bull markets, buy pullbacks to S3/S4 in uptrend; in bear markets, sell rallies to R3/R4 in downtrend.
Daily EMA50 filter ensures trend alignment. Volume confirms institutional participation.
Weekly pivots change slowly, reducing whipsaw. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly high, low, close from prior completed week
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point calculation
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = r3 + (weekly_high - weekly_low)
    s4 = s3 - (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe (using prior week's levels)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for trend filter
    ema_50_daily = df_daily['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume filter (>1.3x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_daily_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or trend reverses
            if close[i] <= s3_aligned[i] or close[i] < ema_50_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 or trend reverses
            if close[i] >= r3_aligned[i] or close[i] > ema_50_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long setup: pullback to S3/S4 in uptrend with volume
            if (close[i] >= s3_aligned[i] and close[i] <= s4_aligned[i] and
                close[i] > ema_50_daily_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short setup: rally to R3/R4 in downtrend with volume
            elif (close[i] <= r3_aligned[i] and close[i] >= r4_aligned[i] and
                  close[i] < ema_50_daily_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals