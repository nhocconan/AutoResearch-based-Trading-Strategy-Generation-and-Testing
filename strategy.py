#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot from previous week (Monday start)
    # Calculate weekly OHLC from daily data
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values  # Previous week high
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values    # Previous week low
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1).values  # Previous week close
    
    # Pivot point = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Support and resistance levels
    R1 = 2 * pivot - weekly_low
    S1 = 2 * pivot - weekly_high
    R2 = pivot + (weekly_high - weekly_low)
    S2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: above 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or
            np.isnan(R2_6h[i]) or np.isnan(S2_6h[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.3 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot R1
            if (close[i] > donchian_high[i] and 
                close[i] > R1_6h[i] and
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly pivot S1
            elif (close[i] < donchian_low[i] and 
                  close[i] < S1_6h[i] and
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot S1 or Donchian low
            if close[i] < S1_6h[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot R1 or Donchian high
            if close[i] > R1_6h[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals