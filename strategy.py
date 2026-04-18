#!/usr/bin/env python3
"""
1d Weekly Pivot Breakout with Volume Confirmation and Trend Filter
Hypothesis: Weekly pivot levels act as strong support/resistance on daily charts.
Breakouts above R1 or below S1 with volume confirmation and weekly trend alignment
capture institutional moves while avoiding false breakouts. Designed for 1d timeframe
to work in both bull and bear markets by filtering with weekly trend and volume.
Target: 10-25 trades/year to minimize fee drag on higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points from prior week
    # Using weekly high, low, close from previous week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly high, low, close from previous week (already completed)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align to daily timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Weekly trend filter: EMA34 on weekly close
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Daily volume filter: 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for weekly data and volume
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(weekly_ema34_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_trend = weekly_ema34_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume and weekly uptrend
            if price > r1_aligned[i] and vol_ok and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and weekly downtrend
            elif price < s1_aligned[i] and vol_ok and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns to pivot or weekly trend breaks down
            if price < pivot_aligned[i] or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to pivot or weekly trend breaks up
            if price > pivot_aligned[i] or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_With_Volume_and_Trend_Filter"
timeframe = "1d"
leverage = 1.0