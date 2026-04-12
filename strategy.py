#!/usr/bin/env python3
"""
6h_1w_1d_Weekly_Pivot_Direction_v1
Hypothesis: Use weekly pivot levels (from 1w) to determine long-term direction, and 1d pivot levels for entry signals.
In uptrend (price above weekly pivot), buy at 1d S1/S2 with reversal signals; in downtrend, sell at 1d R1/R2.
Uses volume confirmation to avoid false breaks. Designed for low trade frequency (15-25/year) by requiring
alignment between weekly trend and daily pivot rejection. Works in bull/bear via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Weekly_Pivot_Direction_v1"
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
    
    # === WEEKLY PIVOT (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high, low, close from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    pp_1w = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3.0
    r1_1w = 2 * pp_1w - low_1w[:-1]
    s1_1w = 2 * pp_1w - high_1w[:-1]
    r2_1w = pp_1w + (high_1w[:-1] - low_1w[:-1])
    s2_1w = pp_1w - (high_1w[:-1] - low_1w[:-1])
    
    # Prepend NaN for first week (no previous week)
    pp_1w = np.concatenate([[np.nan], pp_1w])
    r1_1w = np.concatenate([[np.nan], r1_1w])
    s1_1w = np.concatenate([[np.nan], s1_1w])
    r2_1w = np.concatenate([[np.nan], r2_1w])
    s2_1w = np.concatenate([[np.nan], s2_1w])
    
    # Align weekly pivot to 6h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === DAILY PIVOT (entry signals) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points (using previous day's data)
    pp_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    r1_1d = 2 * pp_1d - low_1d[:-1]
    s1_1d = 2 * pp_1d - high_1d[:-1]
    r2_1d = pp_1d + (high_1d[:-1] - low_1d[:-1])
    s2_1d = pp_1d - (high_1d[:-1] - low_1d[:-1])
    
    # Prepend NaN for first day
    pp_1d = np.concatenate([[np.nan], pp_1d])
    r1_1d = np.concatenate([[np.nan], r1_1d])
    s1_1d = np.concatenate([[np.nan], s1_1d])
    r2_1d = np.concatenate([[np.nan], r2_1d])
    s2_1d = np.concatenate([[np.nan], s2_1d])
    
    # Align daily pivot to 6h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume confirmation (24-period average for 6h = ~6 days)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Weekly trend: price above/below weekly pivot
        above_weekly_pp = close[i] > pp_1w_aligned[i]
        below_weekly_pp = close[i] < pp_1w_aligned[i]
        
        # Entry conditions: fade at daily S1/S2 in uptrend, R1/R2 in downtrend
        long_setup = (
            above_weekly_pp and  # weekly uptrend
            (close[i] <= s1_1d_aligned[i] or close[i] <= s2_1d_aligned[i]) and  # at or below daily S1/S2
            vol_confirm
        )
        
        short_setup = (
            below_weekly_pp and  # weekly downtrend
            (close[i] >= r1_1d_aligned[i] or close[i] >= r2_1d_aligned[i]) and  # at or above daily R1/R2
            vol_confirm
        )
        
        # Exit conditions: reverse when price crosses weekly pivot
        exit_long = below_weekly_pp  # close below weekly pivot
        exit_short = above_weekly_pp  # close above weekly pivot
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals