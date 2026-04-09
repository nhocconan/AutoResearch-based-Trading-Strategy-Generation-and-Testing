#!/usr/bin/env python3
# 6h_donchian_weekly_pivot_volume_v1
# Hypothesis: 6h strategy using Donchian channel breakout (20-period) aligned with weekly pivot point direction and volume confirmation.
# Enters long when price breaks above Donchian upper band AND price is above weekly pivot (bullish bias) with volume spike (>1.5x 20-period avg).
# Enters short when price breaks below Donchian lower band AND price is below weekly pivot (bearish bias) with volume spike.
# Uses discrete position sizing (±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Works in bull/bear by using weekly pivot as dynamic bias filter and Donchian breakouts for momentum entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (high + low + close) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe (completed weekly candle only)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate Donchian channel (20-period) on 6h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume spike detection: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower band
            if close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band
            if close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper AND above weekly pivot (bullish bias) with volume spike
            if (close[i] > donchian_upper[i]) and \
               (close[i] > pivot_1w_aligned[i]) and \
               (vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower AND below weekly pivot (bearish bias) with volume spike
            elif (close[i] < donchian_lower[i]) and \
                 (close[i] < pivot_1w_aligned[i]) and \
                 (vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals