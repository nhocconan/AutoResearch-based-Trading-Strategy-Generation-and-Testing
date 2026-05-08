#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour trend-following with 1-day volatility filter and session timing.
# Uses 4h Donchian breakout for trend direction, 1d ATR ratio to filter low volatility,
# and restricts entries to 08-20 UTC session. Designed to capture trending moves
# while avoiding choppy periods. Target: 60-150 total trades over 4 years.

name = "1h_DonchianTrend_1dATRFilter_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for Donchian channels (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for ATR filter (volatility regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period ATR on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr2])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Current 1d ATR as percentage of price (normalized volatility)
    atr_ratio = atr_14 / close_1d
    # Use 50-period median of ATR ratio to define normal volatility
    atr_median = pd.Series(atr_ratio).rolling(window=50, min_periods=50).median().values
    # Only trade when volatility is above 60% of median (avoid extremely low vol)
    vol_filter = atr_ratio > (atr_median * 0.6)
    
    # Align indicators to 1h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter.astype(float))
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(vol_filter_aligned[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian high with vol filter
            if close[i] > high_20_aligned[i] and vol_filter_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian low with vol filter
            elif close[i] < low_20_aligned[i] and vol_filter_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low or vol filter fails
            if close[i] < low_20_aligned[i] or vol_filter_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high or vol filter fails
            if close[i] > high_20_aligned[i] or vol_filter_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals