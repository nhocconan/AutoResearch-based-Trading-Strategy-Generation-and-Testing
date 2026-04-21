#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_Volume_Confirmation_v1
Hypothesis: On 6h timeframe, go long when price breaks above 20-period Donchian high
with weekly pivot direction bullish (price > weekly pivot) and volume > 1.5x 20-period average.
Go short when price breaks below 20-period Donchian low with weekly pivot direction bearish
(price < weekly pivot) and volume > 1.5x average. Exit when price crosses back through
the 20-period Donchian midpoint. Uses weekly pivot for regime filter to avoid counter-trend
trades and volume confirmation to avoid false breakouts. Target: 15-30 trades/year per symbol.
Works in bull/bear by following higher timeframe (weekly) regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for pivot direction
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Previous week's OHLC for weekly pivot point
    prev_high_w = np.roll(high_w, 1)
    prev_low_w = np.roll(low_w, 1)
    prev_close_w = np.roll(close_w, 1)
    prev_high_w[0] = np.nan
    prev_low_w[0] = np.nan
    prev_close_w[0] = np.nan
    
    # Weekly pivot point: (H + L + C) / 3
    pp_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    pp_w_aligned = align_htf_to_ltf(prices, df_w, pp_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly pivot not ready
        if np.isnan(pp_w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high = prices['high'].iloc[i]
        low = prices['low'].iloc[i]
        close = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate 20-period Donchian channels
        if i >= 20:
            lookback_high = prices['high'].iloc[i-20:i].max()
            lookback_low = prices['low'].iloc[i-20:i].min()
            donchian_high = lookback_high
            donchian_low = lookback_low
            donchian_mid = (donchian_high + donchian_low) / 2
            
            # Volume filter: current volume > 1.5 * 20-period average
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            donchian_high = np.nan
            donchian_low = np.nan
            donchian_mid = np.nan
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above Donchian high + weekly pivot bullish + volume
            if (not np.isnan(donchian_high) and high > donchian_high and 
                close > pp_w_aligned[i] and volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + weekly pivot bearish + volume
            elif (not np.isnan(donchian_low) and low < donchian_low and 
                  close < pp_w_aligned[i] and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below Donchian midpoint
            if not np.isnan(donchian_mid) and close < donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above Donchian midpoint
            if not np.isnan(donchian_mid) and close > donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_Volume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0