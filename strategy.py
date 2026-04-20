#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction filter
# - Donchian(20) breakout on 6h for directional entry
# - Weekly pivot levels (from 1w data) determine long/short bias
# - Only go long when price > weekly pivot point, short when price < weekly pivot point
# - Volume confirmation to avoid false breakouts
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # Pivot Point = (High + Low + Close) / 3
    pp = (high_1w + low_1w + close_1w) / 3
    # Resistance 1 = (2 * Pivot) - Low
    r1 = (2 * pp) - low_1w
    # Support 1 = (2 * Pivot) - High
    s1 = (2 * pp) - high_1w
    # Resistance 2 = Pivot + (High - Low)
    r2 = pp + (high_1w - low_1w)
    # Support 2 = Pivot - (High - Low)
    s2 = pp - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_1w, pp)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate Donchian channels on 6h
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Donchian(20) upper and lower bands
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ok = volume_6h > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close_6h[i] > donch_high[i-1]  # Break above prior period high
        breakout_down = close_6h[i] < donch_low[i-1]  # Break below prior period low
        
        # Weekly pivot bias
        price_above_pivot = close_6h[i] > pp_6h[i]
        price_below_pivot = close_6h[i] < pp_6h[i]
        
        if position == 0:
            # Long entry: upward breakout + price above weekly pivot + volume confirmation
            if breakout_up and price_above_pivot and vol_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: downward breakout + price below weekly pivot + volume confirmation
            elif breakout_down and price_below_pivot and vol_ok[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly pivot or reverse Donchian breakout
            if close_6h[i] < pp_6h[i] or close_6h[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly pivot or reverse Donchian breakout
            if close_6h[i] > pp_6h[i] or close_6h[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_DirectionFilter_Volume"
timeframe = "6h"
leverage = 1.0