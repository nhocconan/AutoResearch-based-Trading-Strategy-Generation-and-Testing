#!/usr/bin/env python3
"""
6h_OrderFlow_Imbalance_Momentum
Hypothesis: Combines volume-weighted price action with momentum to detect institutional order flow.
Uses 1-day VWAP deviation and 6-hour momentum for confluence. Designed for low trade frequency
(15-25 trades/year) to minimize fee burn while capturing sustained moves in both bull and bear
markets by requiring alignment between short-term momentum and institutional volume bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align VWAP to 6h timeframe (using previous day's VWAP)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 6-hour momentum (rate of change over 4 periods = 24h)
    momentum = np.zeros(n)
    momentum[4:] = (close[4:] - close[:-4]) / close[:-4]
    
    # Calculate volume imbalance ratio (buying vs selling pressure)
    # Using close location within the bar's range
    close_location = np.zeros(n)
    ranges = high - low
    mask = ranges > 0
    close_location[mask] = (close[mask] - low[mask]) / ranges[mask]
    close_location[~mask] = 0.5  # When range is zero, assume middle
    
    # Volume-weighted close location (positive = buying pressure, negative = selling)
    vol_weighted_cl = close_location * (2 * volume - np.roll(volume, 1))  # Approximate delta volume
    vol_weighted_cl[0] = 0
    
    # Smooth the volume pressure signal
    vol_pressure = pd.Series(vol_weighted_cl).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(momentum[i]) or
            np.isnan(vol_pressure[i])):
            signals[i] = 0.0
            continue
        
        # Determine volume pressure bias
        buying_pressure = vol_pressure[i] > 0.1
        selling_pressure = vol_pressure[i] < -0.1
        
        # Momentum conditions
        mom_bullish = momentum[i] > 0.015  # >1.5% momentum
        mom_bearish = momentum[i] < -0.015  # <-1.5% momentum
        
        # VWAP deviation
        above_vwap = close[i] > vwap_1d_aligned[i]
        below_vwap = close[i] < vwap_1d_aligned[i]
        
        # Entry conditions: momentum + volume pressure + VWAP alignment
        long_entry = mom_bullish and buying_pressure and above_vwap
        short_entry = mom_bearish and selling_pressure and below_vwap
        
        # Exit conditions: momentum divergence or VWAP reversion
        long_exit = (momentum[i] < -0.005) or (close[i] < vwap_1d_aligned[i] * 0.998)
        short_exit = (momentum[i] > 0.005) or (close[i] > vwap_1d_aligned[i] * 1.002)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_OrderFlow_Imbalance_Momentum"
timeframe = "6h"
leverage = 1.0