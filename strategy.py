#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above 6h Donchian upper (20) AND weekly pivot shows bullish bias (price > weekly pivot) AND volume > 1.5x 20-period average.
Short when price breaks below 6h Donchian lower (20) AND weekly pivot shows bearish bias (price < weekly pivot) AND volume > 1.5x 20-period average.
Exit when price crosses 6h Donchian midpoint (middle of upper/lower channel).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Weekly pivot provides higher-timeframe structure to avoid counter-trend trades in ranging markets.
Donchian channels adapt to volatility, working in both bull and bear markets by capturing breakouts with confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for weekly pivot - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (standard: (H+L+C)/3)
    weekly_pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
    
    # Calculate 6h Donchian channels (20-period)
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_midpoint = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20)  # Ensure warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_midpoint[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly pivot bullish AND volume spike
            if (price > donchian_upper[i] and 
                price > weekly_pivot_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND weekly pivot bearish AND volume spike
            elif (price < donchian_lower[i] and 
                  price < weekly_pivot_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian midpoint
            if position == 1 and price < donchian_midpoint[i]:
                exit_signal = True
            elif position == -1 and price > donchian_midpoint[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1wPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0