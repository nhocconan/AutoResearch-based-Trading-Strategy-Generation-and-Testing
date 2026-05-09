#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot Points
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s1_1w = 2 * pivot_1w - high_1w
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Use previous week's levels (shift by 1 to avoid look-ahead)
    pivot_1w_prev = np.roll(pivot_1w, 1)
    r1_1w_prev = np.roll(r1_1w, 1)
    r2_1w_prev = np.roll(r2_1w, 1)
    s1_1w_prev = np.roll(s1_1w, 1)
    s2_1w_prev = np.roll(s2_1w, 1)
    pivot_1w_prev[0] = np.nan
    r1_1w_prev[0] = np.nan
    r2_1w_prev[0] = np.nan
    s1_1w_prev[0] = np.nan
    s2_1w_prev[0] = np.nan
    
    # Align to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w_prev)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w_prev)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w_prev)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w_prev)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w_prev)
    
    # Weekly trend: EMA21 on weekly close
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume and above weekly EMA21
            if (price > r2_1w_aligned[i] and vol_filter[i] and price > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume and below weekly EMA21
            elif (price < s2_1w_aligned[i] and vol_filter[i] and price < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below pivot (mean reversion)
            if price < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above pivot (mean reversion)
            if price > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals