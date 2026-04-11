#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily pivot-based strategy with weekly trend filter.
# Uses weekly pivot levels (CPR) for trend direction and daily pivot levels for entries.
# Enters long when price tests daily pivot support in bullish weekly trend.
# Enters short when price tests daily pivot resistance in bearish weekly trend.
# Volume filter confirms institutional participation. Designed for 7-25 trades/year on 1d.
# Weekly trend filter reduces whipsaw in sideways markets and improves win rate.

name = "1d_1w_pivot_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly CPR (Central Pivot Range)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Weekly pivot points
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    bc_1w = (prev_high_1w + prev_low_1w) / 2  # Bottom Central
    tc_1w = pivot_1w + (pivot_1w - bc_1w)     # Top Central
    
    # Weekly trend: price above TC = bullish, below BC = bearish
    weekly_trend_bull = close_1w > tc_1w
    weekly_trend_bear = close_1w < bc_1w
    
    # Align weekly levels and trend to daily
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    bc_1w_aligned = align_htf_to_ltf(prices, df_1w, bc_1w)
    tc_1w_aligned = align_htf_to_ltf(prices, df_1w, tc_1w)
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull)
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Daily pivot points
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    r2_1d = pivot_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pivot_1d - (prev_high_1d - prev_low_1d)
    
    # Align daily levels to daily timeframe (no shift needed as we use previous day's data)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or
            np.isnan(bc_1w_aligned[i]) or np.isnan(tc_1w_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or
            np.isnan(weekly_trend_bull_aligned[i]) or np.isnan(weekly_trend_bear_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Determine weekly trend direction
        is_bullish_week = weekly_trend_bull_aligned[i]
        is_bearish_week = weekly_trend_bear_aligned[i]
        
        # Enter long when price tests daily S1 in bullish weekly trend
        long_entry = (low[i] <= s1_1d_aligned[i] and vol_filter and is_bullish_week)
        
        # Enter short when price tests daily R1 in bearish weekly trend
        short_entry = (high[i] >= r1_1d_aligned[i] and vol_filter and is_bearish_week)
        
        # Exit when price reaches daily pivot or opposite S1/R1
        exit_long = (position == 1 and 
                    (high[i] >= pivot_1d_aligned[i] or 
                     low[i] <= s1_1d_aligned[i]))  # Exit long if hits pivot or breaks S1
        exit_short = (position == -1 and 
                     (low[i] <= pivot_1d_aligned[i] or 
                      high[i] >= r1_1d_aligned[i]))  # Exit short if hits pivot or breaks R1
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals