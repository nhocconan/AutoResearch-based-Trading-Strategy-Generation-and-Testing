#!/usr/bin/env python3
name = "6h_Aroon_PivotBreak_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Get 1d data for Aroon and Pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Aroon (25-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Aroon Up: periods since highest high
    aroon_up = np.full(len(high_1d), np.nan)
    for i in range(25, len(high_1d)):
        highest_high_idx = np.argmax(high_1d[i-25:i+1])
        aroon_up[i] = ((25 - highest_high_idx) / 25) * 100
    
    # Aroon Down: periods since lowest low
    aroon_down = np.full(len(low_1d), np.nan)
    for i in range(25, len(low_1d)):
        lowest_low_idx = np.argmin(low_1d[i-25:i+1])
        aroon_down[i] = ((25 - lowest_low_idx) / 25) * 100
    
    # Aroon Oscillator
    aroon_osc = aroon_up - aroon_down
    
    # Align Aroon Oscillator to 6h
    aroon_osc_aligned = align_htf_to_ltf(prices, df_1d, aroon_osc)
    
    # Calculate Daily Pivot Points (Standard)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    r3 = high_prev + 2 * (pivot - low_prev)
    s3 = low_prev - 2 * (high_prev - pivot)
    
    # Align pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 1.8x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(100, 25, 24, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(aroon_osc_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        weekly_uptrend = close > ema_50_1w_aligned[i]
        weekly_downtrend = close < ema_50_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Aroon bullish (up > down) + break above R2 + weekly uptrend + volume
            if (aroon_osc_aligned[i] > 0 and 
                close[i] > r2_aligned[i] and 
                weekly_uptrend and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Aroon bearish (down > up) + break below S2 + weekly downtrend + volume
            elif (aroon_osc_aligned[i] < 0 and 
                  close[i] < s2_aligned[i] and 
                  weekly_downtrend and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Aroon turns bearish OR price breaks below S1
            if (aroon_osc_aligned[i] < 0 or close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Aroon turns bullish OR price breaks above R1
            if (aroon_osc_aligned[i] > 0 or close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combining Aroon oscillator (trend strength) with daily pivot breakouts
# and weekly trend filter creates a robust system for 6x timeframe. Aroon filters
# for genuine trend strength, pivot R2/S2 breakouts capture institutional moves,
# and weekly EMA50 ensures alignment with higher timeframe trend. This should
# work in both bull and bear markets by capturing strong trending moves while
# avoiding choppy periods. Target: 15-25 trades/year (60-100 total over 4 years).