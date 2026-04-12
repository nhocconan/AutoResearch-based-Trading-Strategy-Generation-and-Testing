#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_With_Volume_Filter_v1
Hypothesis: Use weekly Camarilla pivot levels (from 1w data) on 1d chart.
Trade long when price touches S1 level with volume > 1.5x average, short when touches R1 level.
Only trade in direction of 1w EMA20 trend to avoid counter-trend whipsaws.
Targets 10-30 trades/year to minimize fee flood. Works in bull (trend pullbacks to S1) and bear (fades at R1).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_With_Volume_Filter_v1"
timeframe = "1d"
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
    
    # Weekly data for Camarilla pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === WEEKLY CAMARILLA PIVOT LEVELS ===
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and ranges
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    range_ = weekly_high - weekly_low
    
    # Camarilla levels: S1, S2, S3, S4 and R1, R2, R3, R4
    # S1 = C - (H-L)*1.1/12
    # R1 = C + (H-L)*1.1/12
    s1 = pivot - 1.1 * range_ / 12.0
    r1 = pivot + 1.1 * range_ / 12.0
    
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    
    # === WEEKLY EMA20 TREND FILTER ===
    ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d = align_htf_to_ltf(prices, df_1w, ema20)
    
    # === VOLUME FILTER (20-period average) ===
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
        if (np.isnan(s1_1d[i]) or np.isnan(r1_1d[i]) or 
            np.isnan(ema20_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema20_1d[i]
        
        # Entry conditions: price touches S1 (long) or R1 (short) with volume
        # Use high for S1 touch, low for R1 touch
        long_entry = (low[i] <= s1_1d[i]) and vol_confirm and trend_up
        short_entry = (high[i] >= r1_1d[i]) and vol_confirm and not trend_up
        
        # Exit conditions: reverse signal or price returns to weekly pivot
        pivot_1d = align_htf_to_ltf(prices, df_1w, pivot)
        long_exit = not long_entry or close[i] >= pivot_1d[i]
        short_exit = not short_entry or close[i] <= pivot_1d[i]
        
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