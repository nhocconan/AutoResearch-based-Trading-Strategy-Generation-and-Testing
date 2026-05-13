#!/usr/bin/env python3
"""
6h_Donchian20_1dPivotDirection_Volume
Hypothesis: On 6-hour bars, breakouts of 20-period Donchian channels with volume confirmation and aligned 1-day pivot direction (price above/below daily pivot) capture trending moves while avoiding whipsaws. Pivot direction filters breakouts to trade with higher-timeframe bias, improving win rate in both bull and bear markets. Target 25-35 trades/year per symbol.
"""

name = "6h_Donchian20_1dPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily pivot points: P = (H + L + C)/3
    pivot_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot_1d_values = pivot_1d.values
    
    # Align daily pivot to 6h timeframe (wait for daily close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_values)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: price breaks above Donchian high, volume confirmation, price above daily pivot (bullish bias)
            if (close[i] > high_max[i] and 
                volume_filter[i] and 
                close[i] > pivot_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low, volume confirmation, price below daily pivot (bearish bias)
            elif (close[i] < low_min[i] and 
                  volume_filter[i] and 
                  close[i] < pivot_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low OR volume drops
            if (close[i] < low_min[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high OR volume drops
            if (close[i] > high_max[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals