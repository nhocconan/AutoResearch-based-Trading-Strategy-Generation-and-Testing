#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirmation_v1
Hypothesis: On 6h timeframe, trade breakouts of 20-period Donchian channels only when aligned with weekly pivot bias (price above/below weekly pivot) and 1d EMA50 trend, with volume confirmation (>2x median). Weekly pivot provides structural bias from higher timeframe, Donchian breakouts capture momentum, and volume confirmation reduces false breakouts. Designed for low frequency (target 15-30 trades/year) to work in both bull and bear markets by requiring multiple confluences.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get HTF data: 1w for weekly pivot, 1d for EMA trend
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot calculation (using prior week OHLC)
    # Pivot = (H + L + C) / 3
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2x median volume (50-period)
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    # Align HTF indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly pivot (needs 2 weeks), EMA50 (50), Donchian (20), volume median (50)
    start_idx = max(50, 50, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_median[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        weekly_pivot_val = weekly_pivot_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Volume confirmation: volume > 2x median
        volume_confirmed = volume_val > 2.0 * vol_median_val
        
        if position == 0:
            # Long: break above Donchian high with weekly pivot bias (above pivot) and uptrend
            long_signal = (close_val > donchian_high_val) and \
                          (close_val > weekly_pivot_val) and \
                          uptrend and \
                          volume_confirmed
            
            # Short: break below Donchian low with weekly pivot bias (below pivot) and downtrend
            short_signal = (close_val < donchian_low_val) and \
                           (close_val < weekly_pivot_val) and \
                           downtrend and \
                           volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close below Donchian low (mean reversion) or loss of weekly pivot bias
            if close_val < donchian_low_val or close_val < weekly_pivot_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above Donchian high or loss of weekly pivot bias
            if close_val > donchian_high_val or close_val > weekly_pivot_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0