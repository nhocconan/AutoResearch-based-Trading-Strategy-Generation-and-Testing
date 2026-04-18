#!/usr/bin/env python3
"""
12h_WickReversal_WickRatio_VolumeFilter
Hypothesis: Price rejection at long wicks (bullish/bearish) signals reversals. Uses upper/lower wick ratio > 0.6 with volume confirmation to avoid false signals. Works in both bull/bear markets by capturing exhaustion moves. Timeframe 12h reduces trade frequency to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate wick ratios
    body = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    total_range = high - low
    
    # Avoid division by zero
    upper_wick_ratio = np.where(total_range > 0, upper_wick / total_range, 0)
    lower_wick_ratio = np.where(total_range > 0, lower_wick / total_range, 0)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: bullish rejection (long lower wick) with volume spike
            if lower_wick_ratio[i] > 0.6 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: bearish rejection (long upper wick) with volume spike
            elif upper_wick_ratio[i] > 0.6 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: reversal signal or after 8 bars
            if bars_since_entry >= 8 or upper_wick_ratio[i] > 0.6:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: reversal signal or after 8 bars
            if bars_since_entry >= 8 or lower_wick_ratio[i] > 0.6:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WickReversal_WickRatio_VolumeFilter"
timeframe = "12h"
leverage = 1.0