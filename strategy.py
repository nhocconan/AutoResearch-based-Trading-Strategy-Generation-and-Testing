#!/usr/bin/env python3
name = "6h_Weekly_Pivot_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 3:
        return np.zeros(n)
    
    # Calculate weekly high, low, close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot levels (standard formula)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for trend direction
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation (6h volume > 1.5x 24-period average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > vol_ma_24[i] * 1.5
        
        if position == 0:
            # Mean reversion at S1/S2 with bullish daily trend
            if vol_ok and close[i] <= s1_aligned[i] and ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Mean reversion at R1/R2 with bearish daily trend
            elif vol_ok and close[i] >= r1_aligned[i] and ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
            # Breakout continuation at S3/R3 with trend alignment
            elif vol_ok and close[i] <= s3_aligned[i] and ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            elif vol_ok and close[i] >= r3_aligned[i] and ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit at pivot or when price reaches R1
            if close[i] >= pivot_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit at pivot or when price reaches S1
            if close[i] <= pivot_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot mean reversion with daily trend filter on 6h timeframe
# - Uses weekly pivot points (S1/S2/R1/R2) for mean reversion entries
# - Daily EMA20 trend filter ensures trades align with higher timeframe momentum
# - Volume confirmation (1.5x average) filters low-quality signals
# - Works in both bull and bear markets via trend-adaptive mean reversion
# - Breakout continuation at S3/R3 for strong trending moves
# - Exit at pivot level or opposite support/resistance
# - Position size 0.25 targets ~50-150 trades over 4 years (12-37/year)
# - Novel: Weekly pivot mean reversion with trend filter not recently tried on 6h
# - Pivot levels provide institutional support/resistance with clear structure