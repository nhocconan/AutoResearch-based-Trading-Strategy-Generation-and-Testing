#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w pivot direction filter and volume confirmation
# Donchian breakouts capture momentum in trending markets. Weekly pivot provides HTF bias:
# - Price above weekly pivot (PP) = bullish bias, look for long breakouts
# - Price below weekly pivot = bearish bias, look for short breakouts
# Volume confirmation filters false breakouts. Designed for 12-37 trades/year to minimize fee drag.
# Works in bull markets via upside breakouts and in bear markets via downside breakouts.

name = "6h_Donchian20_1wPivot_Direction_VolumeConfirmation"
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
    
    # Get 1w data for pivot calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # Resistance 1 (R1) = (2 * PP) - Low
    # Support 1 (S1) = (2 * PP) - High
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pp_1w) - low_1w
    s1_1w = (2 * pp_1w) - high_1w
    
    # Align weekly pivot to 6h timeframe (wait for completed 1w bar)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND price above weekly PP AND volume spike
            if (close[i] > highest_high_20[i] and 
                close[i] > pp_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND price below weekly PP AND volume spike
            elif (close[i] < lowest_low_20[i] and 
                  close[i] < pp_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR price falls below weekly PP
            if close[i] < lowest_low_20[i] or close[i] < pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR price rises above weekly PP
            if close[i] > highest_high_20[i] or close[i] > pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals