#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot confirmation and volume filter.
# Long when price breaks above Donchian(20) high AND price > 1d weekly pivot R1 AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND price < 1d weekly pivot S1 AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian breakouts capture momentum, 1d weekly pivot provides institutional reference levels,
# volume confirmation ensures participation. Designed to work in both bull (breakout longs) and bear (breakdown shorts).
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Weekly Pivot Points (using prior 1d bar) ===
    # Calculate pivot points from previous 1d bar
    prev_high = pd.Series(high_1d).shift(1)
    prev_low = pd.Series(low_1d).shift(1)
    prev_close = pd.Series(close_1d).shift(1)
    
    # Weekly pivot = (prev_high + prev_low + prev_close) / 3
    weekly_pivot = (prev_high + prev_low + prev_close) / 3
    # R1 = (2 * weekly_pivot) - prev_low
    r1 = (2 * weekly_pivot) - prev_low
    # S1 = (2 * weekly_pivot) - prev_high
    s1 = (2 * weekly_pivot) - prev_high
    
    # Align 1d weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 20 for volume MA, 1 for pivot)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to weekly pivot or volume spike ends
            if price <= weekly_pivot_aligned[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to weekly pivot or volume spike ends
            if price >= weekly_pivot_aligned[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > R1 AND volume spike
            if price > highest_high[i] and price > r1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND price < S1 AND volume spike
            elif price < lowest_low[i] and price < s1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1dWeeklyPivot_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0