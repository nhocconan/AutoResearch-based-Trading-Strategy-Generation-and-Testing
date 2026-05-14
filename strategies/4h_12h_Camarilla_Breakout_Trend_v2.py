#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Trend_v2
Hypothesis: 4h Candlestick close above/below daily Camarilla R4/S4 levels with 12h EMA(21) trend filter and volume confirmation. Designed for low trade frequency (20-50/year) by requiring strong breakouts, trend alignment, and volume surge. Works in bull/bear via EMA trend filter and mean-reversion exit at daily pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Breakout_Trend_v2"
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
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Daily Camarilla levels (R4/S4 for stronger breakouts)
    r4_1d = close_1d + range_1d * 1.5
    s4_1d = close_1d - range_1d * 1.5
    
    # === 12H EMA(21) FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    if len(close_12h) >= 21:
        ema_21_12h = np.zeros_like(close_12h)
        ema_21_12h[0] = close_12h[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close_12h)):
            ema_21_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_21_12h[i-1]
    else:
        ema_21_12h = np.full_like(close_12h, np.nan)
    
    # Align daily and 12h data to 4h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
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
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_21_12h_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2.0x average (adjusted for 4h)
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below 12h EMA(21)
        price_above_ema = close[i] > ema_21_12h_aligned[i]
        price_below_ema = close[i] < ema_21_12h_aligned[i]
        
        # Breakout entries at daily S4/R4 with volume and trend filters
        long_setup = (close[i] > r4_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s4_1d_aligned[i]) and vol_confirm and price_below_ema
        
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