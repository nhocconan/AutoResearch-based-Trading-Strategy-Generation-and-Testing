#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend (price above/below weekly pivot) with volume confirmation (>2.0x average) captures strong directional moves. Weekly pivot acts as regime filter (bull/bear/range). Discrete sizing (0.30) and close-based exits (price retests Donchian level) minimize false signals. Designed for 6h timeframe to avoid overtrading while maintaining edge in both bull and bear markets via trend filter. Target 15-30 trades/year to minimize fee drag.
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot (standard: (H+L+C)/3)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h (wait for completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # ATR(14) for volatility (used in Donchian and volume spike)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation (24-period SMA = 6h * 4 = 1 day)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.30
    
    # Warmup: max of Donchian(20), volume(24)
    start_idx = max(20, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        pivot_val = weekly_pivot_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        # Skip if any data not ready
        if (np.isnan(pivot_val) or np.isnan(avg_vol) or np.isnan(upper) or 
            np.isnan(lower)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Determine regime from weekly pivot
        bullish_regime = close_val > pivot_val
        bearish_regime = close_val < pivot_val
        
        # Long: price CLOSES above Donchian upper in bullish regime with volume
        long_condition = (close_val > upper) and bullish_regime and volume_confirmed
        # Short: price CLOSES below Donchian lower in bearish regime with volume
        short_condition = (close_val < lower) and bearish_regime and volume_confirmed
        
        # Exit: price retests broken Donchian level
        long_exit = (position == 1 and close_val <= upper)
        short_exit = (position == -1 and close_val >= lower)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0