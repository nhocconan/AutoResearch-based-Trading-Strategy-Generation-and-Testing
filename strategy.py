#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout aligned with weekly pivot direction (price above/below weekly pivot) with volume confirmation (>1.5x average volume). Uses discrete sizing (0.25) to minimize fees. Weekly pivot provides structural bias from higher timeframe, Donchian captures breakouts, volume confirms validity. Works in bull/bear by following weekly pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Donchian and volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter (weekly pivot needs daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week (using 5 trading days approx)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # We'll approximate with rolling 5-day window on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close using 5-period rolling on daily data
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (1 bar delay for completed 1d bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Load 1d data for Donchian calculation (using daily high/low)
    # Donchian(20) on 6h: we need 20 periods of 6h data, but we'll use 1d data for structure
    # Actually, let's calculate Donchian on 6h data directly
    # But we need to load 6h data? No, we use prices for 6h
    
    # Calculate Donchian(20) on 6h prices
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian, 20 for volume, 5 for weekly pivot)
    start_idx = max(20, 20, 5)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        wp = weekly_pivot_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(dh) or np.isnan(dl) or np.isnan(avg_vol) or np.isnan(wp)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above Donchian(20) high with price above weekly pivot and volume confirmation
        long_condition = (close_val > dh) and (close_val > wp) and volume_confirmed
        # Short logic: price breaks below Donchian(20) low with price below weekly pivot and volume confirmation
        short_condition = (close_val < dl) and (close_val < wp) and volume_confirmed
        
        # Exit logic: price returns to weekly pivot (mean reversion to pivot)
        exit_long = close_val < wp
        exit_short = close_val > wp
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm"
timeframe = "6h"
leverage = 1.0