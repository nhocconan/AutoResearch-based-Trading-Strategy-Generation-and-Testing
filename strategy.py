#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Retest_v1
Hypothesis: On 12h timeframe, long when price retests weekly Camarilla H3 during 1d uptrend with volume confirmation;
short when price retests weekly Camarilla L3 during 1d downtrend with volume confirmation.
Exit on retest of opposite H4/L4 levels. Uses weekly structure for major trend, daily for intermediate trend,
and 12h for entry timing. Designed for low trade frequency (15-25/year) by requiring multiple confluence factors.
Works in bull/bear via 1d trend filter and mean-reversion exit at weekly Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Retest_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY CAMARILLA LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla pivot levels
    close_prev = close_1w  # using same week's close as approximation
    range_1w = high_1w - low_1w
    
    h5 = close_prev + (range_1w * 1.1 / 2)
    h4 = close_prev + (range_1w * 1.1)
    h3 = close_prev + (range_1w * 1.1 / 4)
    l3 = close_prev - (range_1w * 1.1 / 4)
    l4 = close_prev - (range_1w * 1.1)
    l5 = close_prev - (range_1w * 1.1 / 2)
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA(20) for trend
    ema_20 = np.zeros_like(close_1d)
    ema_20[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_20[i] = (close_1d[i] * 0.0952) + (ema_20[i-1] * 0.9048)  # alpha = 2/(20+1)
    
    # Daily trend: up when price > EMA20, down when price < EMA20
    trend_up = close_1d > ema_20
    trend_down = close_1d < ema_20
    
    # Align weekly Camarilla levels to 12h
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h4_1w_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_1w_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Align daily trend to 12h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    # Volume average (24-period for 12h = ~12 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or 
            np.isnan(h4_1w_aligned[i]) or np.isnan(l4_1w_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Entry conditions: price retesting weekly Camarilla H3/L3 with trend alignment
        # Long: price crosses above H3 during uptrend
        long_setup = (close[i] > h3_1w_aligned[i]) and (close[i-1] <= h3_1w_aligned[i-1]) and \
                     trend_up_aligned[i] > 0.5 and vol_confirm
        # Short: price crosses below L3 during downtrend
        short_setup = (close[i] < l3_1w_aligned[i]) and (close[i-1] >= l3_1w_aligned[i-1]) and \
                      trend_down_aligned[i] > 0.5 and vol_confirm
        
        # Exit conditions: price retesting opposite H4/L4 levels (mean reversion)
        exit_long = (close[i] < l4_1w_aligned[i]) and (close[i-1] >= l4_1w_aligned[i-1])
        exit_short = (close[i] > h4_1w_aligned[i]) and (close[i-1] <= h4_1w_aligned[i-1])
        
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