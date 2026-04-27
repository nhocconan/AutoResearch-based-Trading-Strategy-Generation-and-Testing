#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (R3/S3 levels) with volume confirmation and daily EMA(34) trend filter.
# Enters long when price breaks above S3 with volume, short when breaks below R3 with volume.
# Designed for ~15-25 trades/year by requiring significant breakouts (weekly R3/S3) rather than minor S1/R1 levels.
# Works in bull/bear: buys support breaks, sells resistance breaks.
# Uses strict volume filter (volume > 2x 30-period average) to avoid false breakouts.
# Exit when price returns to pivot or trend changes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week OHLC)
    # Need to aggregate daily to weekly
    # Create weekly OHLC from daily data
    n_days = len(high_1d)
    weeks = n_days // 7
    if weeks < 2:
        return np.zeros(n)
    
    # Reshape to weeks (discard incomplete week)
    high_1d_trim = high_1d[:weeks*7]
    low_1d_trim = low_1d[:weeks*7]
    close_1d_trim = close_1d[:weeks*7]
    
    high_1d_trim = high_1d_trim.reshape(weeks, 7)
    low_1d_trim = low_1d_trim.reshape(weeks, 7)
    close_1d_trim = close_1d_trim.reshape(weeks, 7)
    
    high_weekly = np.max(high_1d_trim, axis=1)
    low_weekly = np.min(low_1d_trim, axis=1)
    close_weekly = close_1d_trim[:, -1]  # last day of week
    
    # Calculate weekly pivot points (using prior week OHLC)
    high_prev = np.roll(high_weekly, 1)
    low_prev = np.roll(low_weekly, 1)
    close_prev = np.roll(close_weekly, 1)
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
    
    # Expand weekly values to daily (each value repeats for 7 days)
    pivot_daily = np.repeat(pivot, 7)
    r3_daily = np.repeat(r3, 7)
    s3_daily = np.repeat(s3, 7)
    
    # Trim to original length
    pivot_daily = pivot_daily[:n_days]
    r3_daily = r3_daily[:n_days]
    s3_daily = s3_daily[:n_days]
    
    # Align daily pivots to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_daily)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_daily)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_daily)
    
    # Daily trend: price above/below daily EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.0 x 30-period average (6h) for significance
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need pivots (1), daily EMA (34), volume MA (30)
    start_idx = max(1, 34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter (strict)
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filters
        daily_bullish = price > ema_34_1d_aligned[i]
        daily_bearish = price < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above S3 with volume and daily bullish
            if price > s3_aligned[i] and vol_filter and daily_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below R3 with volume and daily bearish
            elif price < r3_aligned[i] and vol_filter and daily_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below pivot or daily trend turns bearish
            if price < pivot_aligned[i] or not daily_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above pivot or daily trend turns bullish
            if price > pivot_aligned[i] or not daily_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_S3R3_Volume_Trend"
timeframe = "6h"
leverage = 1.0