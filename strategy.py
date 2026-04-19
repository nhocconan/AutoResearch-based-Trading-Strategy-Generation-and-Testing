#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeBreakout
Hypothesis: Weekly pivot levels act as strong support/resistance. 
- In ranging markets: price tends to revert from weekly R3/S3 (fade)
- In trending markets: breakout beyond weekly R4/S4 signals continuation
- Uses 1d timeframe to calculate weekly pivots (more stable than intraday)
- Volume confirmation filters false breakouts
- Works in bull/bear via adaptive logic: fade in range, breakout in trend
- Target: 50-150 total trades over 4 years (12-37/year)
"""

name = "6h_WeeklyPivot_RangeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_ohlc

def calculate_weekly_pivots(high, low, close):
    """Calculate weekly pivot points from prior week's OHLC"""
    # Typical price
    pp = (high + low + close) / 3.0
    
    # Support and resistance levels
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivots on daily data (using prior week's close)
    # We need to align the weekly pivot calculation to weekly boundaries
    # For simplicity, we'll calculate pivots from daily OHLC and assume
    # they represent the weekly levels (this is an approximation)
    # In practice, we'd want true weekly data, but 1d is acceptable proxy
    
    # Calculate pivots using rolling window of 5 days (1 week)
    high_5d = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close).rolling(window=5, min_periods=5).last().values
    
    # Calculate pivots for each point using prior week's data
    pp = np.full_like(close, np.nan)
    r1 = np.full_like(close, np.nan)
    r2 = np.full_like(close, np.nan)
    r3 = np.full_like(close, np.nan)
    r4 = np.full_like(close, np.nan)
    s1 = np.full_like(close, np.nan)
    s2 = np.full_like(close, np.nan)
    s3 = np.full_like(close, np.nan)
    s4 = np.full_like(close, np.nan)
    
    for i in range(5, len(close)):
        if not (np.isnan(high_5d[i]) or np.isnan(low_5d[i]) or np.isnan(close_5d[i])):
            pp[i], r1[i], r2[i], r3[i], r4[i], s1[i], s2[i], s3[i], s4[i] = \
                calculate_weekly_pivots(high_5d[i], low_5d[i], close_5d[i])
    
    # Align weekly pivot levels to 6h timeframe
    pp_6h = align_ltf_to_ohlc(prices, df_1d, pp)
    r1_6h = align_ltf_to_ohlc(prices, df_1d, r1)
    r2_6h = align_ltf_to_ohlc(prices, df_1d, r2)
    r3_6h = align_ltf_to_ohlc(prices, df_1d, r3)
    r4_6h = align_ltf_to_ohlc(prices, df_1d, r4)
    s1_6h = align_ltf_to_ohlc(prices, df_1d, s1)
    s2_6h = align_ltf_to_ohlc(prices, df_1d, s2)
    s3_6h = align_ltf_to_ohlc(prices, df_1d, s3)
    s4_6h = align_ltf_to_ohlc(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # Choppy market detection: use price position relative to pivots
    # If price is between S3 and R3, we're in a range -> mean reversion
    # If price breaks beyond S4 or R4, we're in a trend -> breakout
    in_range = (close > s3_6h) & (close < r3_6h)
    breakout_long = close > r4_6h
    breakout_short = close < s4_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 5)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entry conditions
            if in_range[i]:
                # In range: fade at R3/S3 levels
                if close[i] <= r3_6h[i] and close[i-1] > r3_6h[i-1]:
                    # Price rejected at R3 -> short
                    if volume_confirm[i]:
                        signals[i] = -0.25
                        position = -1
                elif close[i] >= s3_6h[i] and close[i-1] < s3_6h[i-1]:
                    # Price rejected at S3 -> long
                    if volume_confirm[i]:
                        signals[i] = 0.25
                        position = 1
            else:
                # Trending: look for breakouts
                if breakout_long[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif breakout_short[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long exit conditions
            exit_condition = False
            # Exit if price returns to pivot point (mean reversion)
            if close[i] <= pp_6h[i]:
                exit_condition = True
            # Exit if price reaches R2 and shows weakness
            elif close[i] >= r2_6h[i] and close[i] < close[i-1]:
                exit_condition = True
            # Exit on volume divergence
            elif not volume_confirm[i] and close[i] < close[i-1]:
                exit_condition = True
                
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit conditions
            exit_condition = False
            # Exit if price returns to pivot point (mean reversion)
            if close[i] >= pp_6h[i]:
                exit_condition = True
            # Exit if price reaches S2 and shows strength
            elif close[i] <= s2_6h[i] and close[i] > close[i-1]:
                exit_condition = True
            # Exit on volume divergence
            elif not volume_confirm[i] and close[i] > close[i-1]:
                exit_condition = True
                
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals