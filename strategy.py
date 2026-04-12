#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_Trend_v5
Hypothesis: Daily Camarilla breakout with volume confirmation and daily EMA trend filter on 4h timeframe.
Trades only in direction of higher timeframe trend to avoid whipsaw. Target: 20-50 trades/year.
Works in bull/bear via trend filter - only takes longs in uptrend, shorts in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_Trend_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Daily Camarilla levels
    r3_1d = close_1d + range_1d * 1.1
    s3_1d = close_1d - range_1d * 1.1
    
    # === DAILY EMA(21) FOR TREND FILTER ===
    if len(close_1d) >= 21:
        ema_21_1d = np.zeros_like(close_1d)
        ema_21_1d[0] = close_1d[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close_1d)):
            ema_21_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_21_1d[i-1]
    else:
        ema_21_1d = np.full_like(close_1d, np.nan)
    
    # Align daily data to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume average (20-period for 4h = ~3.3 days) for confirmation
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
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below daily EMA(21)
        price_above_ema = close[i] > ema_21_1d_aligned[i]
        price_below_ema = close[i] < ema_21_1d_aligned[i]
        
        # Breakout entries at daily S3/R3 with volume and trend filters
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