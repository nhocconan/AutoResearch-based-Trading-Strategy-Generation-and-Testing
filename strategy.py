#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour time-weighted average price (TWAP) with 1-day volume-weighted VWAP trend filter
# Long when 6h price > 6h TWAP and 1d VWAP rising, short when 6h price < 6h TWAP and 1d VWAP falling
# Uses volume-weighted average price for institutional trend detection
# TWAP provides dynamic support/resistance based on volume distribution
# Targets 50-150 total trades over 4 years (12-37/year) with disciplined entries
# Works in both bull and bear markets by following institutional volume flow

name = "6h_TWAP_VWAP_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_vals = vwap_1d.values
    
    # Calculate daily VWAP slope (trend)
    vwap_slope = np.diff(vwap_1d_vals, prepend=vwap_1d_vals[0])
    vwap_rising = vwap_slope > 0
    vwap_falling = vwap_slope < 0
    
    # Align VWAP trend to 6h timeframe
    vwap_rising_aligned = align_htf_to_ltf(prices, df_1d, vwap_rising)
    vwap_falling_aligned = align_htf_to_ltf(prices, df_1d, vwap_falling)
    
    # Calculate 6-period TWAP (time-weighted average price)
    # Typical price weighted by time (equal weight per bar) - equivalent to VWAP with unit volume
    typical_price = (high + low + close) / 3
    twap = pd.Series(typical_price).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 6  # warmup for TWAP
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(twap[i]) or np.isnan(vwap_rising_aligned[i]) or 
            np.isnan(vwap_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        twap_val = twap[i]
        vwap_rising_val = vwap_rising_aligned[i]
        vwap_falling_val = vwap_falling_aligned[i]
        
        if position == 0:
            # Enter long: price above TWAP and daily VWAP rising
            if price > twap_val and vwap_rising_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price below TWAP and daily VWAP falling
            elif price < twap_val and vwap_falling_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below TWAP or VWAP stops rising
            if price < twap_val or not vwap_rising_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above TWAP or VWAP stops falling
            if price > twap_val or not vwap_falling_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals