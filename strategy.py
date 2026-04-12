#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_Trend_v1
Hypothesis: Weekly Camarilla pivot breakouts with weekly EMA(21) trend filter and volume confirmation on 1d timeframe. 
Designed for low trade frequency (7-25/year) to avoid drag. Works in bull/bear via trend filter.
Entry: Price breaks weekly R3/S3 with volume > 2x average and price above/below weekly EMA(21).
Exit: Price returns to weekly pivot (mean reversion).
Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_Trend_v1"
timeframe = "1d"
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
    
    # Weekly pivot calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla levels
    r3_1w = close_1w + range_1w * 1.1
    s3_1w = close_1w - range_1w * 1.1
    
    # === WEEKLY EMA(21) FOR TREND FILTER ===
    if len(close_1w) >= 21:
        ema_21_1w = np.zeros_like(close_1w)
        ema_21_1w[0] = close_1w[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close_1w)):
            ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    else:
        ema_21_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly data to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Volume average (20-period for 1d = ~1 month) for confirmation
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
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below weekly EMA(21)
        price_above_ema = close[i] > ema_21_1w_aligned[i]
        price_below_ema = close[i] < ema_21_1w_aligned[i]
        
        # Breakout entries at weekly S3/R3 with volume and trend filters
        long_setup = (close[i] > r3_1w_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s3_1w_aligned[i]) and vol_confirm and price_below_ema
        
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