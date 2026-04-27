#!/usr/bin/env python3
"""
6h_Donchian_20_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: Donchian(20) breakouts aligned with weekly pivot point direction and volume confirmation capture high-probability trend moves. Weekly pivot provides institutional bias; volume filters false breakouts. Works in bull/bear by following pivot-derived trend direction.
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
    
    # Get weekly data for pivot points and trend
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    pivot_w = (high_w + low_w + close_w) / 3.0
    
    # Weekly trend: price above/below pivot
    weekly_bullish = close_w > pivot_w
    weekly_bearish = close_w < pivot_w
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align weekly indicators to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian (20), weekly data (1), volume avg (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot = pivot_w_aligned[i]
        weekly_bull = weekly_bullish_aligned[i]
        weekly_bear = weekly_bearish_aligned[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: Donchian breakout above resistance + weekly bullish + volume
            if weekly_bull and vol_close and close_val > dch_high:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short: Donchian breakdown below support + weekly bearish + volume
            elif weekly_bear and vol_conf and close_val < dch_low:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: price returns to weekly pivot or opposite Donchian break
            if close_val <= pivot or close_val < dch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price returns to weekly pivot or opposite Donchian break
            if close_val >= pivot or close_val > dch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0