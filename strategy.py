#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d VWAP trend and volume spike confirmation.
# Long when price breaks above Donchian high (20) AND price > 1d VWAP AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian low (20) AND price < 1d VWAP AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (midline).
# This strategy captures breakouts with institutional volume confirmation and trend alignment.
# VWAP provides a dynamic equilibrium level, superior to simple moving averages for intraday context.
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d VWAP trend direction.

name = "4h_Donchian_20_1dVWAP_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1-day VWAP (volume-weighted average price)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align VWAP to 4h timeframe (constant throughout the day)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Donchian channel (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Sufficient warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, price > VWAP, volume filter
            long_cond = (close[i] > high_max[i]) and (close[i] > vwap_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below Donchian low, price < VWAP, volume filter
            short_cond = (close[i] < low_min[i]) and (close[i] < vwap_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian midline
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian midline
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals