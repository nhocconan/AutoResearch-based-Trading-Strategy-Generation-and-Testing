#!/usr/bin/env python3
"""
Hypothesis: 6-hour Volume-Weighted Average Price (VWAP) with 1-week high-low channel filter.
Long when price > VWAP and above weekly low, short when price < VWAP and below weekly high.
VWAP provides intraday fair value; weekly channel filters extremes and prevents counter-trend trades.
Designed for low trade frequency by requiring both VWAP deviation and channel position.
Works in ranging markets (mean reversion to VWAP) and trends (filter prevents false signals).
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
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, 0.0)
    
    # Load 1-week data for high-low channel - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly highest high and lowest low (expanding window)
    high_1w_expanding = np.maximum.accumulate(high_1w)
    low_1w_expanding = np.minimum.accumulate(low_1w)
    
    # Align to 6h timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w_expanding)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w_expanding)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 for VWAP calculation
        # Skip if VWAP or weekly data not ready
        if vwap_denominator[i] == 0 or np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above VWAP and above weekly low (not in extreme high zone)
            if close[i] > vwap[i] and close[i] > low_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below VWAP and below weekly high (not in extreme low zone)
            elif close[i] < vwap[i] and close[i] < high_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below VWAP
                if close[i] < vwap[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above VWAP
                if close[i] > vwap[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_VWAP_WeeklyChannel_Filter"
timeframe = "6h"
leverage = 1.0