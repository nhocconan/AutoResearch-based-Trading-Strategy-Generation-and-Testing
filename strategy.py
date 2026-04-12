#!/usr/bin/env python3
"""
1d_1w_Camarilla_Reversal_v2
Hypothesis: On 1d timeframe, buy reversals at Camarilla L3/L4 with 1w uptrend filter and volume confirmation,
sell reversals at H3/H4 with 1w downtrend and volume confirmation. Exit at opposite L3/H3 levels.
Uses weekly trend filter to avoid counter-trend trades. Designed for low trade frequency
(10-25/year) by requiring multiple confluence factors. Works in bull/bear via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Reversal_v2"
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
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for Camarilla calculation
    close_prev = np.concatenate([[close_1d[0]], close_1d[:-1]])
    range_1d = high_1d - low_1d
    
    h3 = close_prev + (range_1d * 1.1 / 4)
    h4 = close_prev + (range_1d * 1.1)
    l3 = close_prev - (range_1d * 1.1 / 4)
    l4 = close_prev - (range_1d * 1.1)
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend
    ema_21 = np.zeros_like(close_1w)
    ema_21[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_21[i] = (close_1w[i] * 2 / (21 + 1)) + (ema_21[i-1] * (21 - 1) / (21 + 1))
    
    # Align data to daily timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume average (20-period) for confirmation
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
        if (np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Entry conditions: reversal at extreme levels with weekly trend
        long_setup = (close[i] <= l3_aligned[i]) and (close[i] > l4_aligned[i]) and vol_confirm and (close[i] > ema_21_aligned[i])
        short_setup = (close[i] >= h3_aligned[i]) and (close[i] < h4_aligned[i]) and vol_confirm and (close[i] < ema_21_aligned[i])
        
        # Exit conditions: reverse at opposite level
        exit_long = close[i] >= h3_aligned[i]
        exit_short = close[i] <= l3_aligned[i]
        
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