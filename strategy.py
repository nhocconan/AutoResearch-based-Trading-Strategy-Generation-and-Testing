#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_Volume
# Hypothesis: Trade weekly pivot breakouts with daily trend and volume confirmation on 6h timeframe.
# Weekly pivots provide strong support/resistance that holds across market regimes.
# Daily trend filter ensures we trade with higher timeframe momentum.
# Volume confirmation filters out false breakouts.
# Designed for low trade frequency (15-30/year) with high win rate in both bull and bear markets.

name = "6h_WeeklyPivot_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot_point - prev_week_low
    s1 = 2 * pivot_point - prev_week_high
    r2 = pivot_point + (prev_week_high - prev_week_low)
    s2 = pivot_point - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot_point - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot_point)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above weekly R3 + daily uptrend + volume spike
            if close[i] > r3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below weekly S3 + daily downtrend + volume spike
            elif close[i] < s3_aligned[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly pivot or trend reversal
            if close[i] <= pivot_point[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot or trend reversal
            if close[i] >= pivot_point[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals