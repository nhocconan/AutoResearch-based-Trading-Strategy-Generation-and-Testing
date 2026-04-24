#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Donchian channels and weekly pivot.
- Donchian breakout: Long when price breaks above 20-period high, short when breaks below 20-period low.
- Weekly pivot filter: Only trade long if price > weekly pivot, short if price < weekly pivot (from 1d data aggregated to weekly).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakouts in downtrend.
- Uses weekly pivot from 1d data to avoid look-ahead and ensure proper alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period) on 1d
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Weekly pivot from 1d data (using prior week's data to avoid look-ahead)
    # Calculate weekly OHLC from daily data
    df_1d_df = pd.DataFrame({
        'open': df_1d['open'].values,
        'high': high_1d,
        'low': low_1d,
        'close': close_1d
    }, index=pd.to_datetime(df_1d['open_time']))
    
    # Resample to weekly (using actual weekly bars, not resampling)
    # We'll use the mtf_data approach: get weekly data directly
    try:
        df_1w = get_htf_data(prices, '1w')
        if len(df_1w) >= 50:
            high_1w = df_1w['high'].values
            low_1w = df_1w['low'].values
            close_1w = df_1w['close'].values
            
            # Weekly pivot: (high + low + close) / 3
            weekly_pivot = (high_1w + low_1w + close_1w) / 3
            
            # Align weekly pivot to 6h
            weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
    except:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Donchian + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout with weekly pivot filter
            if close[i] > donchian_high_20_aligned[i] and close[i] > weekly_pivot_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low_20_aligned[i] and close[i] < weekly_pivot_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or weekly pivot
            if close[i] < donchian_low_20_aligned[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or weekly pivot
            if close[i] > donchian_high_20_aligned[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0