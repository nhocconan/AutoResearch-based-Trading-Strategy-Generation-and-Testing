#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    # For daily data, weekly OHLC is aggregated from 5 trading days
    # We'll use the last 5 days to approximate weekly OHLC
    high_5d = np.full_like(high_1d, np.nan)
    low_5d = np.full_like(low_1d, np.nan)
    close_5d = np.full_like(close_1d, np.nan)
    
    for i in range(4, len(high_1d)):
        high_5d[i] = np.max(high_1d[i-4:i+1])
        low_5d[i] = np.min(low_1d[i-4:i+1])
        close_5d[i] = close_1d[i]
    
    # Shift by 1 to use previous week's data
    high_prev = np.roll(high_5d, 1)
    low_prev = np.roll(low_5d, 1)
    close_prev = np.roll(close_5d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Weekly pivot calculation
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r2 = pivot + (high_prev - low_prev)  # R2 = Pivot + (High - Low)
    s2 = pivot - (high_prev - low_prev)  # S2 = Pivot - (High - Low)
    
    # Align weekly pivots to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Weekly trend: price above/below weekly EMA(34) on daily data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.8 x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot (5), weekly EMA (34), volume MA (20)
    start_idx = max(5, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: price crosses above S2 with volume and above weekly EMA
            if price > s2_aligned[i] and vol_filter and price > ema_34_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: price crosses below R2 with volume and below weekly EMA
            elif price < r2_aligned[i] and vol_filter and price < ema_34_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below pivot or below weekly EMA
            if price < pivot_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above pivot or above weekly EMA
            if price > pivot_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_S2R2_VolumeTrend"
timeframe = "6h"
leverage = 1.0