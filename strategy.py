#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend Filter with Volume Confirmation
Hypothesis: Weekly pivot points act as strong support/resistance levels. 
Breakouts above weekly R3/R4 or below S3/S4 with daily trend alignment and volume 
confirmation capture significant momentum moves. Works in bull/bear via daily trend filter.
Target: 75-150 trades over 4 years (19-38/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly high, low, close for pivot calculation
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    weekly_r4 = weekly_r3 + (weekly_high - weekly_low)
    weekly_s4 = weekly_s3 - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 week for no look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # Load daily data for trend filter (once before loop)
    df_daily = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend filter
    daily_close = df_daily['close'].values
    ema50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_prev = np.roll(ema50_daily, 1)
    ema50_daily_prev[0] = ema50_daily[0]
    ema50_rising = ema50_daily > ema50_daily_prev
    ema50_falling = ema50_daily < ema50_daily_prev
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_daily, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_daily, ema50_falling)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 24-period EMA (4 days worth of 6h bars)
    vol_ema = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For daily EMA50 and weekly pivot stability
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ema[i]) or
            np.isnan(ema50_daily_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below weekly S3 OR stoploss
            if (close[i] <= s3_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above weekly R3 OR stoploss
            if (close[i] >= r3_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: weekly pivot breakout + daily trend + volume
            bull_breakout = close[i] > r4_aligned[i]  # Break above R4 for strong bullish momentum
            bear_breakout = close[i] < s4_aligned[i]  # Break below S4 for strong bearish momentum
            
            bull_entry = bull_breakout and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals