#!/usr/bin/env python3
"""
#100987 - 6h_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
In bull markets: buy breakouts above weekly pivot, sell breakdowns below weekly pivot.
In bear markets: fade false breakouts at weekly pivot levels (mean reversion).
Weekly pivot provides structural support/resistance that works across regimes.
Volume filter ensures breakouts have conviction. Target: 15-25 trades/year.
"""

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
    
    # Get weekly data for pivot calculation (previous week to avoid look-ahead)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    weekly_r1 = weekly_pivot + (weekly_range * 1.1) / 12  # R1 = PP + (H-L)*1.1/12
    weekly_s1 = weekly_pivot - (weekly_range * 1.1) / 12  # S1 = PP - (H-L)*1.1/12
    
    # Align weekly levels to 6h timeframe (previous week's levels)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: Donchian breakout above weekly pivot with volume
        if (close[i] > donchian_high[i] and 
            close[i] > weekly_pivot_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: Donchian breakdown below weekly pivot with volume
        elif (close[i] < donchian_low[i] and 
              close[i] < weekly_pivot_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly pivot (mean reversion)
        elif position == 1 and close[i] < weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0