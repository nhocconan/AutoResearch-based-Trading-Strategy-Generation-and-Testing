#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Trend_v1
Hypothesis: Use daily Camarilla pivot levels as dynamic support/resistance. 
Breakout above R4 or below S4 with volume confirmation and 1-day trend filter (price above/below EMA50).
Only take long when trend up and price breaks R4, short when trend down and price breaks S4.
Targets 20-40 trades per year to minimize fee drag. Works in bull (follow breakouts) and bear (fade false breaks).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's range)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # S4 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.1/4
    # where C = (H+L+CLOSE)/3 of previous day
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Previous day's values for today's pivots
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    prev_high[0] = daily_high[0]  # first bar uses same day
    prev_low[0] = daily_low[0]
    prev_close[0] = daily_close[0]
    
    # Pivot point and range
    pivot = (prev_high + prev_low + prev_close) / 3
    rng = prev_high - prev_low
    
    # Camarilla levels
    R4 = pivot + rng * 1.1 / 2
    R3 = pivot + rng * 1.1 / 4
    S3 = pivot - rng * 1.1 / 4
    S4 = pivot - rng * 1.1 / 2
    
    # Align Camarilla levels to 4h
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA50 for trend filter
    ema50 = np.full(len(daily_close), np.nan)
    if len(daily_close) >= 50:
        alpha = 2 / (50 + 1)
        ema50[0] = daily_close[0]
        for i in range(1, len(daily_close)):
            ema50[i] = alpha * daily_close[i] + (1 - alpha) * ema50[i-1]
    
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(R4_4h[i]) or np.isnan(S4_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: daily close above/below EMA50
        daily_close_val = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_val)
        trend_up = daily_close_aligned[i] > ema50_4h[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Breakout conditions
        breakout_up = high[i] > R4_4h[i] and vol_confirm
        breakout_down = low[i] < S4_4h[i] and vol_confirm
        
        # Entry logic: only follow trend
        long_entry = breakout_up and trend_up
        short_entry = breakout_down and not trend_up
        
        # Exit logic: reverse signal or price returns to pivot
        pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
        long_exit = not breakout_up or close[i] < pivot_4h[i]
        short_exit = not breakout_down or close[i] > pivot_4h[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals