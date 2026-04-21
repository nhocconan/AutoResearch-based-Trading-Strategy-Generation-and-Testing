#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_Plus_Volume
Hypothesis: On 6h, use weekly pivot direction (from 1w) as the primary trend filter, and enter on breakouts from the previous day's high/low with volume confirmation. This combines higher timeframe directional bias (weekly pivot) with short-term breakout logic, aiming to capture momentum moves in both bull and bear markets. The weekly pivot provides a robust trend filter that adapts to changing market regimes, while daily breakouts with volume capture entry timing. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Weekly trend filter: pivot point direction ===
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    # Weekly pivot point
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Trend is bullish if price above weekly pivot, bearish if below
    trend_w = pivot_w  # we'll use this as the reference level
    trend_w_aligned = align_htf_to_ltf(prices, df_1w, trend_w)
    
    # === Daily breakout levels: previous day's high/low ===
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    # Previous day's high/low (shifted by 1 to avoid look-ahead)
    prev_high_d = np.roll(high_d, 1)
    prev_low_d = np.roll(low_d, 1)
    # First day has no previous, set to NaN
    prev_high_d[0] = np.nan
    prev_low_d[0] = np.nan
    # Align to 6h timeframe
    prev_high_d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_d)
    prev_low_d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_d)
    
    # === Volume confirmation: 20-period volume average on 6h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(trend_w_aligned[i]) or
            np.isnan(prev_high_d_aligned[i]) or
            np.isnan(prev_low_d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_pivot = trend_w_aligned[i]
        prev_high = prev_high_d_aligned[i]
        prev_low = prev_low_d_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above previous day's high + volume spike > 1.5 + price above weekly pivot
            if (price_close > prev_high and 
                vol_spike > 1.5 and 
                price_close > weekly_pivot):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below previous day's low + volume spike > 1.5 + price below weekly pivot
            elif (price_close < prev_low and 
                  vol_spike > 1.5 and 
                  price_close < weekly_pivot):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to weekly pivot
            if position == 1 and price_close < weekly_pivot:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_Direction_Plus_Volume"
timeframe = "6h"
leverage = 1.0