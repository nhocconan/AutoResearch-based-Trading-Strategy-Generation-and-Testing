#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# - Weekly pivot points determine primary trend direction (only trade long above PP, short below PP)
# - Daily volume > 1.5x 20-day average confirms institutional participation
# - 60-period EMA on 6s as secondary trend filter
# - Designed for 60-100 total trades over 4 years (15-25/year) to minimize fee drag
# - Works in both bull/bear via pivot-based directional filtering
name = "6h_Donchian_WeeklyPivot_Volume_EMA60"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot points: (H+L+C)/3
    # Pivot calculation requires complete weekly candles - use shift(1) to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align pivot to 6s timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 60-period EMA on 6s for secondary trend filter
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # 6s Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(ema_60[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + above weekly pivot + above EMA60 + volume
            if (close[i] > donch_high[i] and 
                close[i] > pivot_aligned[i] and 
                close[i] > ema_60[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + below weekly pivot + below EMA60 + volume
            elif (close[i] < donch_low[i] and 
                  close[i] < pivot_aligned[i] and 
                  close[i] < ema_60[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR below weekly pivot
            if close[i] < donch_low[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR above weekly pivot
            if close[i] > donch_high[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals