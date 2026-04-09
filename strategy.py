#!/usr/bin/env python3
# 6h_donchian_12h_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakout with 12h Camarilla pivot direction filter and volume confirmation.
# Enters long when price breaks above 6h Donchian upper band AND price > 12h H3 level with volume spike.
# Enters short when price breaks below 6h Donchian lower band AND price < 12h L3 level with volume spike.
# Uses 12h Camarilla levels as dynamic support/resistance to filter breakouts.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to avoid fee drag.
# Works in bull/bear by using Donchian breakouts with pivot level confirmation.
# Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_12h_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian(20) breakout
    high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h HTF data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    # Pivot point = (high + low + close) / 3
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    h3 = pivot + (range_12h * 1.1 / 4)
    l3 = pivot - (range_12h * 1.1 / 4)
    h4 = pivot + (range_12h * 1.1 / 2)
    l4 = pivot - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (completed 12h candle only)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_12h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, l4)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_6h[i]) or np.isnan(low_6h[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below 6h Donchian lower band
            if close[i] < low_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above 6h Donchian upper band
            if close[i] > high_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 6h Donchian upper band AND price > 12h H3 level with volume spike
            if (close[i] > high_6h[i]) and \
               (close[i] > h3_aligned[i]) and \
               (vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 6h Donchian lower band AND price < 12h L3 level with volume spike
            elif (close[i] < low_6h[i]) and \
                 (close[i] < l3_aligned[i]) and \
                 (vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals