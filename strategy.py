# USING RESEARCH NOTES: Experiment #150858
# Daily timeframe with weekly HTF - Donchian breakout + volume confirmation + weekly trend filter
# Target: 30-100 total trades over 4 years (7-25/year)
# Strategy designed to work in both bull and bear markets via trend filter and breakout logic

name = "1d_Donchian20_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (updated once per week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Daily price data for Donchian channels
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs weekly EMA20
        uptrend_1w = close[i] > ema20_1w_aligned[i]
        downtrend_1w = close[i] < ema20_1w_aligned[i]
        
        # Volume filter: current daily volume > 1.3x 20-day MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.3
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend with volume
            if close[i] > donch_high[i] and uptrend_1w and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in downtrend with volume
            elif close[i] < donch_low[i] and downtrend_1w and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend fails
            if close[i] < donch_low[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend fails
            if close[i] > donch_high[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3