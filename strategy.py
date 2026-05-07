#!/usr/bin/env python3
name = "6h_WeeklyPivot_Donchian_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader's method)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = 3*P - 2*L, S4 = 3*H - 2*L
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r4 = 3 * pivot - 2 * weekly_low
    s4 = 3 * weekly_high - 2 * weekly_low
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Daily trend filter: EMA 34
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channel (20-period) for breakout confirmation
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema_34[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price breaks above weekly R4 with volume and daily uptrend
            if close[i] > r4_aligned[i] and close[i] > donchian_upper[i] and vol_condition and ema_34[i] > ema_34[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 with volume and daily downtrend
            elif close[i] < s4_aligned[i] and close[i] < donchian_lower[i] and vol_condition and ema_34[i] < ema_34[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below weekly pivot or trend reverses
            if close[i] < pivot_aligned[i] or ema_34[i] < ema_34[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above weekly pivot or trend reverses
            if close[i] > pivot_aligned[i] or ema_34[i] > ema_34[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Weekly Pivot Donchian Breakout with Daily Trend Filter
# - Uses weekly pivot points (R4/S4) as significant support/resistance levels
# - Breakouts occur when price closes beyond weekly R4 (for longs) or S4 (for shorts)
# - Requires Donchian(20) breakout in same direction for momentum confirmation
# - Volume filter (1.5x average) ensures genuine interest
# - Daily EMA34 trend filter ensures alignment with intermediate trend
# - Exits when price returns to weekly pivot or trend reverses
# - Works in bull markets (buying R4 breakouts in uptrend) and bear markets (selling S4 breakdowns in downtrend)
# - Weekly pivot levels are widely watched by institutions, increasing reliability
# - Combines multiple confirmation layers to reduce false signals
# - Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Weekly timeframe for pivots avoids noise, daily trend for alignment, 6h for execution