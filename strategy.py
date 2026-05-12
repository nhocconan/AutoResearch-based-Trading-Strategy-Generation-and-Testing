# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with weekly pivot point reversal zones + volume confirmation + trend filter.
- Uses weekly pivot points (calculated from prior week) to identify key support/resistance zones
- Long when price bounces off weekly S1/S2 with volume confirmation and weekly uptrend
- Short when price rejects weekly R1/R2 with volume confirmation and weekly downtrend
- Weekly trend filter (EMA34) avoids counter-trend trades
- Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag
"""
name = "6h_WeeklyPivot_Reversal_Volume_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, etc."""
    pivot = (high + low + close) / 3.0
    s1 = 2 * pivot - high
    r1 = 2 * pivot - low
    s2 = pivot - (high - low)
    r2 = pivot + (high - low)
    return pivot, s1, s2, r1, r2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_pivot = np.full_like(weekly_high, np.nan)
    weekly_s1 = np.full_like(weekly_high, np.nan)
    weekly_s2 = np.full_like(weekly_high, np.nan)
    weekly_r1 = np.full_like(weekly_high, np.nan)
    weekly_r2 = np.full_like(weekly_high, np.nan)
    
    for i in range(1, len(weekly_high)):
        p, s1, s2, r1, r2 = calculate_weekly_pivot(
            weekly_high[i-1], weekly_low[i-1], weekly_close[i-1]
        )
        weekly_pivot[i] = p
        weekly_s1[i] = s1
        weekly_s2[i] = s2
        weekly_r1[i] = r1
        weekly_r2[i] = r2
    
    # Align weekly pivot levels to 6h timeframe (using prior week's values)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    
    # Weekly trend filter: EMA34 on weekly close
    ema_34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Volume filter: current volume > 1.8x 20-period average (to avoid noise)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure volume average has enough data
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price near weekly support + volume spike + weekly uptrend
            near_support = (low[i] <= s1_aligned[i] * 1.005 and low[i] >= s2_aligned[i] * 0.995) or \
                           (low[i] <= s2_aligned[i] * 1.005 and low[i] >= s2_aligned[i] * 0.99)
            weekly_uptrend = close[i] > ema_34_aligned[i]
            
            if near_support and vol_filter[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short setup: price near weekly resistance + volume spike + weekly downtrend
            near_resistance = (high[i] >= r1_aligned[i] * 0.995 and high[i] <= r2_aligned[i] * 1.005) or \
                              (high[i] >= r2_aligned[i] * 0.995 and high[i] <= r2_aligned[i] * 1.01)
            weekly_downtrend = close[i] < ema_34_aligned[i]
            
            if near_resistance and vol_filter[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S2 or weekly trend turns down
            if low[i] < s2_aligned[i] * 0.99 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R2 or weekly trend turns up
            if high[i] > r2_aligned[i] * 1.01 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals