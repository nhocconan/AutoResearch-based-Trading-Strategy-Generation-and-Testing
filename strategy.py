#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_v3
Hypothesis: 4-hour strategy using daily Camarilla pivot levels with volume confirmation.
Enters long when price breaks above H3 with volume spike; short when breaks below L3 with volume spike.
Uses fixed position sizing (0.25) to minimize churn and uses simple exit on opposite H3/L3 touch.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag while capturing strong moves.
Focus on breakout quality with volume filter to avoid false signals in chop.
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels using PREVIOUS day's data (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day's previous values to NaN (no data yet)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations using previous day's data
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    h4 = pivot + 1.1 * range_val
    l4 = pivot - 1.1 * range_val
    
    # Align Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average (strict filter)
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Entry conditions: Camarilla H3/L3 breakout with volume confirmation
        long_breakout = close[i] > h3_4h[i] and volume_filter
        short_breakout = close[i] < l3_4h[i] and volume_filter
        
        # Exit conditions: touch opposite H3/L3 level
        long_exit = close[i] < l3_4h[i]
        short_exit = close[i] > h3_4h[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0