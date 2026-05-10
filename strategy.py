#!/usr/bin/env python3
# 6h_Weekly_Pivot_Range_Exhaustion
# Hypothesis: Weekly pivot ranges act as institutional support/resistance zones. Price exhaustion at
# weekly R4/S4 levels (extreme weekly ranges) with volume confirmation and alignment with daily trend
# provides high-probability mean-reversion entries. Works in bull/bear by fading extremes rather than
# chasing trends. Targets 20-50 trades/year via strict weekly level confluence.

name = "6h_Weekly_Pivot_Range_Exhaustion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === WEEKLY PIVOT CALCULATION ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point = (H + L + C)/3
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly range
    weekly_range = weekly_high - weekly_low
    # Support/resistance levels
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + weekly_range
    s2 = pivot - weekly_range
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r4 = weekly_high + 3 * (weekly_high - weekly_low)  # Extreme resistance
    s4 = weekly_low - 3 * (weekly_high - weekly_low)   # Extreme support
    
    # Align weekly levels to 6t
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values  # 24 * 6h = 4d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24) + 1
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long setup: price at or below weekly S4 (extreme support) + daily uptrend + volume
            if close[i] <= s4_aligned[i] and close[i] > ema50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short setup: price at or above weekly R4 (extreme resistance) + daily downtrend + volume
            elif close[i] >= r4_aligned[i] and close[i] < ema50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above daily EMA50 OR reaches weekly pivot
            if close[i] >= ema50_1d_aligned[i] or close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below daily EMA50 OR reaches weekly pivot
            if close[i] <= ema50_1d_aligned[i] or close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals