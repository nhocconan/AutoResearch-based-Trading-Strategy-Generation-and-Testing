#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_Volume_TrendFilter
# Hypothesis: Weekly pivot levels (R1, S1) derived from 1w OHLC provide key support/resistance.
# Breakout above weekly R1 or below S1 with volume confirmation and trend filter (1d EMA50) signals institutional interest.
# In bull markets: R1 breakouts lead to continuation. In bear markets: S1 breaks lead to continuation.
# Volume filters false breakouts, trend filter ensures alignment with higher timeframe momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WeeklyPivot_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (R1, S1) from previous week's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C, H, L are close, high, low of previous week
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    # First value will be invalid (no previous week), handled by alignment
    
    # Calculate weekly pivot R1 and S1
    weekly_r1 = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 12
    weekly_s1 = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 12
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume confirmation and above 1d EMA50 (uptrend)
            if (close[i] > weekly_r1_aligned[i] and volume_confirm[i] and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume confirmation and below 1d EMA50 (downtrend)
            elif (close[i] < weekly_s1_aligned[i] and volume_confirm[i] and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly S1 (reversal) or below 1d EMA50 (trend change)
            if close[i] < weekly_s1_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly R1 (reversal) or above 1d EMA50 (trend change)
            if close[i] > weekly_r1_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals