#!/usr/bin/env python3
"""
Hypothesis: Daily timeframe strategy using weekly pivot point breakout with 1-week trend filter and volume confirmation.
Enters long when price breaks above weekly R1 with above-average volume and weekly uptrend.
Enters short when price breaks below weekly S1 with above-average volume and weekly downtrend.
Uses weekly timeframe for structure and trend, daily for execution to reduce noise.
Designed to work in both bull and bear markets by following the weekly trend and requiring volume confirmation.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
"""

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
    
    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Support and resistance levels
    S1 = (2 * pivot) - high_1w
    S2 = pivot - (high_1w - low_1w)
    S3 = low_1w - 2 * (high_1w - pivot)
    R1 = (2 * pivot) - low_1w
    R2 = pivot + (high_1w - low_1w)
    R3 = high_1w + 2 * (pivot - low_1w)
    
    # Align weekly pivot levels to daily timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (00-24 UTC for daily timeframe)
    # For daily timeframe, we can trade all hours as signals are based on daily close
    # But we'll still use time filtering to avoid low-volume periods if needed
    
    # Warmup: need weekly pivot levels, volume MA, and weekly EMA
    start_idx = max(5, 20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current daily price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_20_1w_aligned[i]
        
        # Current weekly pivot levels
        S1_now = S1_aligned[i]
        R1_now = R1_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Weekly pivot breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above R1 with volume + weekly uptrend
            if price_now > R1_now and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume + weekly downtrend
            elif price_now < S1_now and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or weekly trend turns down
            pivot_now = (high_1w[i//7 if i>=7 else 0] + low_1w[i//7 if i>=7 else 0] + close_1w[i//7 if i>=7 else 0]) / 3.0 if i>=7 else pivot[0]
            # Simpler: use aligned pivot (need to calculate it)
            pivot_aligned = align_htf_to_ltf(prices, df_1w, (high_1w + low_1w + close_1w) / 3.0)
            pivot_now = pivot_aligned[i]
            if price_now <= pivot_now or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or weekly trend turns up
            pivot_aligned = align_htf_to_ltf(prices, df_1w, (high_1w + low_1w + close_1w) / 3.0)
            pivot_now = pivot_aligned[i]
            if price_now >= pivot_now or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyPivotBreakout_1dVolume_1wTrend"
timeframe = "1d"
leverage = 1.0