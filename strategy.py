#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Weekly_Pivot_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data once
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Previous day's pivot points (HLC/3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Weekly pivot levels from prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_1w = prev_high_1w - prev_low_1w
    
    # Pivot support/resistance levels (daily)
    s1_daily = pivot_1d - (range_1d * 1.1 / 12)
    r1_daily = pivot_1d + (range_1d * 1.1 / 12)
    
    # Weekly pivot levels (major support/resistance)
    s1_weekly = pivot_1w - (range_1w * 1.1 / 12)
    r1_weekly = pivot_1w + (range_1w * 1.1 / 12)
    
    # Align to 6h timeframe
    s1_daily_6h = align_htf_to_ltf(prices, df_1d, s1_daily)
    r1_daily_6h = align_htf_to_ltf(prices, df_1d, r1_daily)
    s1_weekly_6h = align_htf_to_ltf(prices, df_1w, s1_weekly)
    r1_weekly_6h = align_htf_to_ltf(prices, df_1w, r1_weekly)
    
    # === Trend filter: 6h EMA50 ===
    ema50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(s1_daily_6h[i]) or np.isnan(r1_daily_6h[i]) or 
            np.isnan(s1_weekly_6h[i]) or np.isnan(r1_weekly_6h[i]) or
            np.isnan(ema50_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above daily S1 and weekly S1, above EMA50, volume confirmation
            long_cond = (close[i] > s1_daily_6h[i] and 
                        close[i] > s1_weekly_6h[i] and
                        close[i] > ema50_6h[i] and
                        volume[i] > vol_ma20[i])
            
            # Short conditions: price below daily R1 and weekly R1, below EMA50, volume confirmation
            short_cond = (close[i] < r1_daily_6h[i] and 
                         close[i] < r1_weekly_6h[i] and
                         close[i] < ema50_6h[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below daily S1 or weekly S1
            exit_cond = (close[i] < s1_daily_6h[i] or 
                        close[i] < s1_weekly_6h[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above daily R1 or weekly R1
            exit_cond = (close[i] > r1_daily_6h[i] or 
                        close[i] > r1_weekly_6h[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot levels act as strong institutional support/resistance.
# In 6h timeframe, we take longs when price holds above both daily S1 and weekly S1
# with EMA50 trend filter and volume confirmation. Shorts when below both daily R1
# and weekly R1. Exits when price breaks these key levels. Designed to work in
# both bull (buying dips to weekly support) and bear (selling rallies to weekly
# resistance) markets. Targets 50-150 trades over 4 years to minimize fee drag.