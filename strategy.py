#!/usr/bin/env python3
name = "6h_WeeklyPivot_RangeBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly pivot points from 1w data (calculated from previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate pivot points: (H + L + C) / 3
    # Use previous week's data (already complete) - no lookahead
    prev_week_high = df_1w['high'].values
    prev_week_low = df_1w['low'].values
    prev_week_close = df_1w['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot)
    
    # Align to 6t timeframe (values change only when new week starts)
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily trend filter (1d EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~18 hours
    
    start_idx = max(20, 1)  # Ensure volume MA and pivot data are valid
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_6h[i]) or 
            np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above weekly R3 with volume in uptrend
            if close[i] > r3_6h[i] and vol_ok[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: break below weekly S3 with volume in downtrend
            elif close[i] < s3_6h[i] and vol_ok[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price returns below pivot OR trend turns down
            if close[i] < pivot_6h[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above pivot OR trend turns up
            if close[i] > pivot_6h[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points act as significant support/resistance levels.
# Breaking above R3 or below S3 with volume confirmation indicates strong momentum.
# The 1d EMA34 filter ensures we only trade in the direction of the daily trend,
# avoiding counter-trend whipsaws. This strategy captures breakouts from weekly
# ranges while using volume to confirm institutional participation. Works in both
# bull and bear markets by following the daily trend direction. Target: 15-35 trades/year.