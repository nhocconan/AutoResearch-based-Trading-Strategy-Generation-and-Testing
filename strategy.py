#!/usr/bin/env python3
name = "6h_StructureBreakout_WeeklyPivot_Trend"
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
    
    # === Weekly pivot points (calculated on daily data, aligned to 6h) ===
    df_1d = get_htf_data(prices, '1d')
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).mean().values
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    r2 = pivot_point + (weekly_high - weekly_low)
    s2 = pivot_point - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot_point - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot_point)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === Trend filter: 6h EMA50 (avoid look-ahead) ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume filter: volume > 1.3 * 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if weekly pivot data not ready (need 5 days of weekly data)
        if np.isnan(pivot_aligned[i]) or np.isnan(ema50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above weekly R2 with volume + above EMA50
            if high[i] > r2_aligned[i] and vol_filter[i] and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S2 with volume + below EMA50
            elif low[i] < s2_aligned[i] and vol_filter[i] and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Break below weekly S1 or trend reversal
            if low[i] < s1_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Break above weekly R1 or trend reversal
            if high[i] > r1_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals