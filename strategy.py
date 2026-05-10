#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction
Hypothesis: Donchian(20) breakout combined with weekly pivot direction and volume confirmation creates robust trend-following signals that work in both bull and bear markets. Weekly pivots provide institutional reference levels, while Donchian channels capture breakouts with volume confirmation reducing false signals.
Timeframe: 6h balances trade frequency (target: 50-150 total trades over 4 years) and signal quality for BTC/ETH.
"""

name = "6h_Donchian20_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    prev_high = np.roll(high_weekly, 1)
    prev_low = np.roll(low_weekly, 1)
    prev_close = np.roll(close_weekly, 1)
    prev_high[0] = high_weekly[0]
    prev_low[0] = low_weekly[0]
    prev_close[0] = close_weekly[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - prev_close)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Get daily data for trend filter (EMA34)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Get 6h data for Donchian channel and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_aligned = align_htf_to_ltf(prices, df_6h, vol_ema20)
    
    # 6h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (1 period) + Donchian (20) + daily EMA (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema34_daily_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ema20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs daily EMA34
        uptrend_daily = close[i] > ema34_daily_aligned[i]
        downtrend_daily = close[i] < ema34_daily_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > vol_ema20_aligned[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume and daily uptrend, above weekly pivot
            if (high[i] > donchian_high[i] and 
                volume_filter and 
                uptrend_daily and 
                close[i] > pivot_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume and daily downtrend, below weekly pivot
            elif (low[i] < donchian_low[i] and 
                  volume_filter and 
                  downtrend_daily and 
                  close[i] < pivot_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend fails
            if low[i] < donchian_low[i] or not uptrend_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend fails
            if high[i] > donchian_high[i] or not downtrend_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals