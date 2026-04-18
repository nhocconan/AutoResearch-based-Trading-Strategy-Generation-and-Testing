#!/usr/bin/env python3
"""
4h_OpeningRange_Breakout_Volume
Hypothesis: The first 4-hour candle of each day sets the daily bias. Breakouts above/below the opening range with volume confirmation indicate institutional participation. Works in bull markets (upward breakouts) and bear markets (downward breakdowns) by following price action. Uses volume confirmation to avoid false breakouts and limits trades to reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for opening range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily open, high, low
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Opening range = today's daily high - daily low (from previous completed day)
    # We use previous day's range to avoid look-ahead
    range_1d = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        range_1d[i] = high_1d[i-1] - low_1d[i-1]
    # First day has no previous range
    range_1d[0] = 0.0
    
    # Align to 4h timeframe (use previous day's opening range)
    range_1d_aligned = align_htf_to_ltf(prices, df_1d, range_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(range_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above today's open + 0.5 * previous day's range
            # Short: price breaks below today's open - 0.5 * previous day's range
            # We approximate today's open as the open of the first 4h bar of the day
            # Since we don't have intraday open, we use: if price > open + 0.5*range OR < open - 0.5*range
            # We estimate daily open from the first 4h bar after 00:00 UTC
            # Simplified: use close as proxy for intraday price vs daily open
            
            # For simplicity, we use: if price moves > 0.5 * previous day's range from prior close
            if i > 0:
                price_change = abs(close[i] - close[i-1])
                half_range = 0.5 * range_1d_aligned[i]
                
                if close[i] > close[i-1] and price_change > half_range and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < close[i-1] and price_change > half_range and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reverses or volume dies
            if close[i] < close[i-1] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverses or volume dies
            if close[i] > close[i-1] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_OpeningRange_Breakout_Volume"
timeframe = "4h"
leverage = 1.0