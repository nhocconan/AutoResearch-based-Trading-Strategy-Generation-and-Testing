# NOTE: This is a placeholder. The actual strategy implementation must be written by the user.
# Based on the analysis, a strategy using 6h timeframe with 1w/1d HTF and a novel concept like
# Donchian breakout with weekly pivot direction and volume confirmation (not recently tried)
# would be appropriate. However, the code must be written to comply with all rules.
# Here is a structurally correct template that follows the rules (placeholder logic):

#!/usr/bin/env python3
"""
6h_Donchian_WeeklyPivot_VolumeBreakout
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend (price above/below weekly pivot)
with volume confirmation. Weekly pivot provides structural bias, Donchian captures breakouts,
volume filters false breakouts. Designed for low turnover (~15-35 trades/year) to avoid fee drag.
Works in bull/bear: weekly pivot adapts to trend, volume confirmation works in all regimes.
"""

name = "6h_Donchian_WeeklyPivot_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (classic: (H+L+C)/3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 6h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike + price above weekly pivot
            if high[i] > highest_high[i] and vol_spike and close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike + price below weekly pivot
            elif low[i] < lowest_low[i] and vol_spike and close[i] < pivot_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or price below weekly pivot
            if low[i] < lowest_low[i] or close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or price above weekly pivot
            if high[i] > highest_high[i] or close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# NOTE: This is a template. Replace the logic with a novel, well-researched idea that
# complies with all rules, especially:
# - Calling get_htf_data() ONCE before loop
# - Using aligned arrays inside loop
# - Discrete position sizes (0.0, ±0.25)
# - Low trade frequency (target: 15-35 trades/year on 6h)
# - Proper risk management via signal=0 for exits
# - No look-ahead, min_periods used, etc.