#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily pivot points with R3/S3 levels and volume confirmation.
# Enters long when price breaks above S3 with volume and price above weekly EMA(50).
# Enters short when price breaks below R3 with volume and price below weekly EMA(50).
# Uses weekly EMA(50) as trend filter and volume > 2.0 x 30-period average as confirmation.
# Designed for 12-37 trades/year by requiring significant breakouts (R3/S3) rather than minor S1/R1 levels.
# Works in bull/bear: buys support breaks, sells resistance breaks.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day OHLC)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    r3 = high_prev + 2 * (pivot - low_prev)
    s3 = low_prev - 2 * (high_prev - pivot)
    
    # Align daily pivots to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly trend: price above/below weekly EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 2.0 x 30-period average (12h) for significance
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need pivots (1), weekly EMA (50), volume MA (30)
    start_idx = max(1, 50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter (strict)
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filters
        weekly_bullish = price > ema_50_1w_aligned[i]
        weekly_bearish = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above S3 with volume and weekly bullish
            if price > s3_aligned[i] and vol_filter and weekly_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below R3 with volume and weekly bearish
            elif price < r3_aligned[i] and vol_filter and weekly_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below pivot or weekly trend turns bearish
            if price < pivot_aligned[i] or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above pivot or weekly trend turns bullish
            if price > pivot_aligned[i] or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Pivot_S3R3_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0