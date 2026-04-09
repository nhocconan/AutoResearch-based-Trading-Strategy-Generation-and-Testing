#!/usr/bin/env python3
# 6h_donchian_1w_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Enters long when price breaks above 6h Donchian upper channel with volume spike and weekly pivot bullish bias.
# Enters short when price breaks below 6h Donchian lower channel with volume spike and weekly pivot bearish bias.
# Uses weekly Camarilla pivot (R4/S4) as trend filter: price > weekly R4 = bullish bias, price < weekly S4 = bearish bias.
# Designed for medium trade frequency (target: 50-150 total trades over 4 years) to balance signal quality and fee drag.
# Works in bull/bear by using weekly pivot levels as dynamic trend filter and Donchian breakouts as momentum signals.
# Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_1w_pivot_volume_v1"
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
    open_time = prices['open_time'].values
    
    # 6h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # 1d HTF data for weekly aggregation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of daily data
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Camarilla pivot levels from daily data
    # Group daily data into weeks (approximate: 5 trading days per week)
    # We'll use rolling window of 5 days to simulate weekly levels
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    pivot_5d = (high_5d + low_5d + close_5d) / 3
    range_5d = high_5d - low_5d
    
    # Weekly Camarilla levels
    # R4 = pivot + (range * 1.1/2)
    # R3 = pivot + (range * 1.1/4)
    # S3 = pivot - (range * 1.1/4)
    # S4 = pivot - (range * 1.1/2)
    r4 = pivot_5d + (range_5d * 1.1 / 2)
    s4 = pivot_5d - (range_5d * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    # First align to 1d, then we'll use the values as weekly filters
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly pivot trend filter
        bullish_bias = close[i] > r4_aligned[i]   # Price above weekly R4 = bullish
        bearish_bias = close[i] < s4_aligned[i]   # Price below weekly S4 = bearish
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower channel
            if close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper channel
            if close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper with volume spike and bullish bias
            if (close[i] > donchian_upper[i]) and \
               (vol_spike[i]) and \
               (bullish_bias):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower with volume spike and bearish bias
            elif (close[i] < donchian_lower[i]) and \
                 (vol_spike[i]) and \
                 (bearish_bias):
                position = -1
                signals[i] = -0.25
    
    return signals