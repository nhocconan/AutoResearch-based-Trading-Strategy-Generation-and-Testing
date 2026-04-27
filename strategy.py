#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week OHLC
    # We'll resample daily data to weekly using simple aggregation on indices
    # Since we don't have actual weekly data, we'll approximate using 5-day rolling
    # For proper weekly pivot, we need actual week boundaries
    # Instead, we'll use monthly pivot concept adapted to weekly: use 5-day prior OHLC
    lookback = 5  # approximate 1 week (5 trading days)
    
    high_prev = np.roll(high_1d, lookback)
    low_prev = np.roll(low_1d, lookback)
    close_prev = np.roll(close_1d, lookback)
    # Set first 'lookback' values to NaN
    high_prev[:lookback] = np.nan
    low_prev[:lookback] = np.nan
    close_prev[:lookback] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    r3 = high_prev + 2 * (pivot - low_prev)
    s3 = low_prev - 2 * (high_prev - pivot)
    
    # Align weekly pivots to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Weekly trend: price above/below weekly EMA(34) - using 5-day EMA as proxy
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.5 x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need pivots (5), weekly EMA (34), volume MA (20)
    start_idx = max(5, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter
        bullish_weekly = price > ema_34_1d_aligned[i]
        bearish_weekly = price < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price crosses above S1 with volume and bullish weekly trend
            if price > s1_aligned[i] and vol_filter and bullish_weekly:
                signals[i] = size
                position = 1
            # Short: price crosses below R1 with volume and bearish weekly trend
            elif price < r1_aligned[i] and vol_filter and bearish_weekly:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below pivot or weekly trend turns bearish
            if price < pivot_aligned[i] or not bullish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above pivot or weekly trend turns bullish
            if price > pivot_aligned[i] or not bearish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_S1R1_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0