#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) and volume confirmation capture sustained moves. Weekly pivot provides higher-timeframe bias, Donchian breakout captures momentum, volume confirms institutional participation. Discrete sizing (0.25) limits fee churn. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for Donchian and weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot from prior week (using daily data)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Approximate using last 5 daily bars for weekly levels
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align all indicators to primary timeframe (6h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Volume confirmation: current volume > 1.8 * 20-period average (6h equivalent)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Donchian(20), weekly pivot (5), volume avg (20)
    start_idx = max(20, 5, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        pivot_val = weekly_pivot_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine bias: price relative to weekly pivot
            bias_long = close_val > pivot_val
            bias_short = close_val < pivot_val
            
            if bias_long:
                # Long bias: long when price breaks above upper Donchian band with volume
                if (close_val > upper_band) and vol_conf:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif bias_short:
                # Short bias: short when price breaks below lower Donchian band with volume
                if (close_val < lower_band) and vol_conf:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (2.0*ATR) or opposite Donchian touch
            # Simple ATR approximation using 6h range
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < lower_band:  # Opposite band touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (2.0*ATR) or opposite Donchian touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > upper_band:  # Opposite band touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0