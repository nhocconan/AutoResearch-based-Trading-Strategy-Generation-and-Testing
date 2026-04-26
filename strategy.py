#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot trend direction (price above/below weekly pivot) and volume confirmation (>1.5x 20-bar average) captures institutional breakouts with controlled frequency. Weekly pivot provides multi-day trend bias that works in both bull and bear markets, Donchian breakouts capture momentum, volume confirms participation. Targets 12-37 trades/year (50-150 over 4 years) on 6h timeframe. Uses discrete sizing (0.25) to minimize fee churn.
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
    
    # Get 1d data for weekly pivot calculation (need 5 days for weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from prior week (using last 5 daily bars)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Use rolling window of 5 for weekly high/low/close
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Donchian(20) on 6h
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (already aligned as we're using 6h data)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # weekly pivot (5*24=120 6h bars approx) and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        pivot_val = weekly_pivot_aligned[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with weekly pivot trend and volume
            # Long: price breaks above Donchian high with price > weekly pivot (uptrend bias)
            long_signal = (high_val > donch_high) and (close_val > pivot_val) and volume_confirm
            # Short: price breaks below Donchian low with price < weekly pivot (downtrend bias)
            short_signal = (low_val < donch_low) and (close_val < pivot_val) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below Donchian low (exit long)
            if low_val < donch_low:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend change: price crosses below weekly pivot (exit long)
            elif close_val < pivot_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above Donchian high (exit short)
            if high_val > donch_high:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend change: price crosses above weekly pivot (exit short)
            elif close_val > pivot_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0