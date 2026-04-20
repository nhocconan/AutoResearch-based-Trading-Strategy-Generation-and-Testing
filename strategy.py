#!/usr/bin/env python3
"""
4h_Turtle_Soup_With_Volume_v1
Concept: 4h Turtle Soup reversal pattern with volume confirmation and trend filter.
- Long: Price breaks below 20-period low, then reverses above it within 3 bars, with volume > 1.5x average and price > EMA50
- Short: Price breaks above 20-period high, then reverses below it within 3 bars, with volume > 1.5x average and price < EMA50
- Exit: Price crosses EMA50 in opposite direction
- Position sizing: 0.25
- Works in bull/bear: EMA50 defines intermediate trend, Turtle Soup captures false breakout reversals, volume confirms conviction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Turtle_Soup_With_Volume_v1"
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
    
    # === 4h: EMA50 trend filter ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: 20-period high/low for breakout detection ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    consecutive_low_break = 0  # consecutive bars below low_20
    consecutive_high_break = 0  # consecutive bars above high_20
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        ema50_val = ema50[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_val) or np.isnan(high_20_val) or np.isnan(low_20_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Track consecutive breaks
        if low_val < low_20_val:
            consecutive_low_break += 1
        else:
            consecutive_low_break = 0
            
        if high_val > high_20_val:
            consecutive_high_break += 1
        else:
            consecutive_high_break = 0
        
        if position == 0:
            # Long setup: price broke below 20-period low, now reversing back above it
            # Must happen within 3 bars of breakdown to be valid Turtle Soup
            if (consecutive_low_break > 0 and consecutive_low_break <= 3 and 
                close_val > low_20_val and  # price moved back above the breakdown level
                vol_ratio_val > 1.5 and     # volume confirmation
                close_val > ema50_val):     # trend filter: above EMA50
                signals[i] = 0.25
                position = 1
            # Short setup: price broke above 20-period high, now reversing back below it
            elif (consecutive_high_break > 0 and consecutive_high_break <= 3 and 
                  close_val < high_20_val and   # price moved back below the breakout level
                  vol_ratio_val > 1.5 and       # volume confirmation
                  close_val < ema50_val):       # trend filter: below EMA50
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA50 (trend change)
            if close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA50 (trend change)
            if close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals