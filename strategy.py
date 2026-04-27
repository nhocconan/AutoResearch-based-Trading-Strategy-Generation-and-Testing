#!/usr/bin/env python3
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
    
    # Get daily data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly pivot points (calculated from previous week)
    high_weekly = df_1d['high'].rolling(5, min_periods=5).max().shift(1)  # Previous week high
    low_weekly = df_1d['low'].rolling(5, min_periods=5).min().shift(1)    # Previous week low
    close_weekly = df_1d['close'].rolling(5, min_periods=5).mean().shift(1)  # Previous week close
    
    # Weekly pivot: P = (H+L+C)/3
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3
    # Weekly R1 and S1
    weekly_r1 = 2 * weekly_pivot - low_weekly
    weekly_s1 = 2 * weekly_pivot - high_weekly
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivots and daily EMA to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_daily)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup period
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: price above weekly pivot, above daily EMA34, with volume spike
            if (price > weekly_pivot_aligned[i] and 
                price > ema34_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot, below daily EMA34, with volume spike
            elif (price < weekly_pivot_aligned[i] and 
                  price < ema34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly pivot or volume drops
            if price < weekly_pivot_aligned[i] or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly pivot or volume drops
            if price > weekly_pivot_aligned[i] or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_1dTrend_WithVolume_v3"
timeframe = "6h"
leverage = 1.0