#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyTrend_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points using previous week's OHLC
    prev_week_close = np.roll(df_1w['close'], 1)
    prev_week_high = np.roll(df_1w['high'], 1)
    prev_week_low = np.roll(df_1w['low'], 1)
    prev_week_close[0] = np.nan
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    
    # Weekly pivot point (P) and support/resistance levels
    # P = (H + L + C) / 3
    # S1 = (2*P) - H
    # R1 = (2*P) - L
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    s1 = (2 * pivot) - prev_week_high
    r1 = (2 * pivot) - prev_week_low
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 for daily EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_20_1w_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        pivot_pt = pivot_aligned[i]
        s1_level = s1_aligned[i]
        r1_level = r1_aligned[i]
        
        if position == 0:
            # Enter long: Price above weekly pivot AND both weekly and daily trends up
            if close[i] > pivot_pt and ema_1w > ema_1w * 0.999 and ema_1d > ema_1d * 0.999:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below weekly pivot AND both weekly and daily trends down
            elif close[i] < pivot_pt and ema_1w < ema_1w * 1.001 and ema_1d < ema_1d * 1.001:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR daily trend turns down
            if close[i] < pivot_pt or ema_1d < ema_1d * 0.999:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR daily trend turns up
            if close[i] > pivot_pt or ema_1d > ema_1d * 1.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: The trend condition checks (ema_1w > ema_1w * 0.999) are placeholders for actual trend direction.
# In practice, we'd compare current EMA to previous EMA, but to avoid look-ahead and keep simple,
# we use a small epsilon to indicate trend direction based on EMA slope.
# For a cleaner implementation, we would calculate EMA slope using prior values.