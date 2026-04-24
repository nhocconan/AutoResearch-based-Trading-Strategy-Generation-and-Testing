#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter and volume confirmation.
- Long when price breaks above Donchian(20) high AND price > weekly pivot (bullish regime) AND volume > 1.5 * 20-period average
- Short when price breaks below Donchian(20) low AND price < weekly pivot (bearish regime) AND volume > 1.5 * 20-period average
- Exit when price reverts to Donchian(20) midpoint OR weekly pivot regime flips
- Uses 6h primary with 1w HTF for pivot regime to capture major trend direction
- Donchian provides objective breakout levels; weekly pivot filters for higher-timeframe bias; volume confirms conviction
- Designed to catch strong trending moves while avoiding counter-trend breakouts in ranging markets
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1w pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 weeks for pivot calculation
        return np.zeros(n)
    
    # Weekly pivot: (Prior week High + Low + Close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align weekly pivot to 6h timeframe (completed weekly pivot only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 20) + 1  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high AND above weekly pivot AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND below weekly pivot AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price reverts to Donchian midpoint OR price falls below weekly pivot
            if close[i] <= donchian_mid[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price reverts to Donchian midpoint OR price rises above weekly pivot
            if close[i] >= donchian_mid[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0