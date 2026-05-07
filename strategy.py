#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Trend
Hypothesis: Price breaking above/below weekly Camarilla pivot levels with daily trend filter and volume confirmation captures momentum moves. Weekly pivots act as strong support/resistance, daily trend filter ensures alignment with higher timeframe momentum, and volume confirmation reduces false signals. Designed for low frequency via 1d timeframe with strict entry criteria.
Target: 30-100 total trades over 4 years.
"""
name = "1d_WeeklyPivot_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high, low, close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla pivot calculations
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    range_val = weekly_high - weekly_low
    
    # Resistance and support levels
    r4 = pivot + (range_val * 1.1)
    r3 = pivot + (range_val * 1.1/2)
    r2 = pivot + (range_val * 1.1/4)
    r1 = pivot + (range_val * 1.1/6)
    s1 = pivot - (range_val * 1.1/6)
    s2 = pivot - (range_val * 1.1/4)
    s3 = pivot - (range_val * 1.1/2)
    s4 = pivot - (range_val * 1.1)
    
    # Align weekly levels to daily timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + daily uptrend + volume
            if close[i] > r3_aligned[i] and close[i] > ema_50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + daily downtrend + volume
            elif close[i] < s3_aligned[i] and close[i] < ema_50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to pivot (mean reversion to mean)
            if position == 1:
                if close[i] <= pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals