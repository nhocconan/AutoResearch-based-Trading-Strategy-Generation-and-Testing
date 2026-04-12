#!/usr/bin/env python3
"""
6h_1d_Price_Action_Breakout_v1
Hypothesis: 6h breakouts above/below daily high/low with volume confirmation and 12h EMA(34) trend filter. 
Works in bull/bear via EMA trend filter and mean-reversion exit at daily pivot.
Target: 12-37 trades/year (50-150 total over 4 years) by requiring strong breakouts with volume and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Price_Action_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR PIVOT LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points (classic)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Key levels: R1, S1, R2, S2, R3, S3
    r1_1d = pivot_1d + range_1d
    s1_1d = pivot_1d - range_1d
    r2_1d = pivot_1d + 2 * range_1d
    s2_1d = pivot_1d - 2 * range_1d
    r3_1d = pivot_1d + 3 * range_1d
    s3_1d = pivot_1d - 3 * range_1d
    
    # === 12H EMA(34) FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    if len(close_12h) >= 34:
        ema_34_12h = np.zeros_like(close_12h)
        ema_34_12h[0] = close_12h[0]
        alpha = 2.0 / (34 + 1)
        for i in range(1, len(close_12h)):
            ema_34_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_34_12h[i-1]
    else:
        ema_34_12h = np.full_like(close_12h, np.nan)
    
    # Align daily and 12h data to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume average (20-period for 6h = ~5 days) for confirmation
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
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2.0x average (adjusted for 6h)
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below 12h EMA(34)
        price_above_ema = close[i] > ema_34_12h_aligned[i]
        price_below_ema = close[i] < ema_34_12h_aligned[i]
        
        # Breakout entries at S3/R3 with volume and trend filters (stronger levels)
        long_setup = (close[i] > r3_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s3_1d_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit when price returns to daily pivot (mean reversion)
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        
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