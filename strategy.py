#!/usr/bin/env python3
"""
4h_1d_Weekly_Pivot_Breakout_v1
Hypothesis: 4h breakouts above/below weekly pivot R2/S2 levels with 1d EMA(20) trend filter and volume confirmation.
Targets weekly pivot levels (stronger than daily) to reduce trades while maintaining edge in bull/bear via trend filter.
Designed for low trade frequency (20-50/year) by requiring significant breakouts above weekly resistance/support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Weekly_Pivot_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation (using prior week's data)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly pivot levels - R2/S2 for stronger breakouts
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # === DAILY EMA(20) FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 20:
        ema_20_1d = np.zeros_like(close_1d)
        ema_20_1d[0] = close_1d[0]
        alpha = 2.0 / (20 + 1)
        for i in range(1, len(close_1d)):
            ema_20_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_20_1d[i-1]
    else:
        ema_20_1d = np.full_like(close_1d, np.nan)
    
    # Align weekly and daily data to 4h timeframe
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume average (20-period for 4h = ~1.3 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2.0x average (adjusted for 4h)
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below 1d EMA(20)
        price_above_ema = close[i] > ema_20_1d_aligned[i]
        price_below_ema = close[i] < ema_20_1d_aligned[i]
        
        # Breakout entries at weekly S2/R2 with volume and trend filters
        long_setup = (close[i] > r2_1w_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s2_1w_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit when price returns to weekly pivot (mean reversion)
        exit_long = close[i] < pivot_1w_aligned[i]
        exit_short = close[i] > pivot_1w_aligned[i]
        
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