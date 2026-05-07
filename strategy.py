#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Weekly_Pivot_Filter
Hypothesis: Use daily KAMA direction for trend bias and weekly pivot support/resistance for entry timing on 12h timeframe. Enter long when price pulls back to weekly S1 in uptrend with volume confirmation, short when price rallies to weekly R1 in downtrend with volume confirmation. Exits when price moves against trend or volume dries up. Designed to work in both bull and bear markets by aligning with daily trend while using weekly structure for mean-reversion entries. Target: 20-40 trades/year to minimize fee drag.
"""

name = "12h_KAMA_Trend_With_Weekly_Pivot_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate daily KAMA (10, 2, 30) for trend filter
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Align daily KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Previous week's OHLC for weekly pivot calculation
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    # Align previous week's OHLC to 12h timeframe
    prev_weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_high)
    prev_weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_low)
    prev_weekly_close_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_close)
    
    # Calculate weekly pivot point and support/resistance levels
    pp = (prev_weekly_high_aligned + prev_weekly_low_aligned + prev_weekly_close_aligned) / 3.0
    r1 = 2 * pp - prev_weekly_low_aligned
    s1 = 2 * pp - prev_weekly_high_aligned
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(kama_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend determination using KAMA
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > kama_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < kama_1d_aligned[i]
        
        if position == 0:
            # Long: price pulls back to S1 in uptrend with volume confirmation
            if (close[i] <= s1[i] * 1.005 and  # Allow small buffer for entry
                close[i] >= s1[i] * 0.995 and
                vol_ratio[i] > 1.8 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price rallies to R1 in downtrend with volume confirmation
            elif (close[i] >= r1[i] * 0.995 and 
                  close[i] <= r1[i] * 1.005 and
                  vol_ratio[i] > 1.8 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price moves below S1 or trend turns down
            if (close[i] < s1[i] * 0.99 or not daily_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price moves above R1 or trend turns up
            if (close[i] > r1[i] * 1.01 or not daily_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals