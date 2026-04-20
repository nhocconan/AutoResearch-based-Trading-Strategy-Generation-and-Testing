#!/usr/bin/env python3
"""
6h_1d_Turtle_Soup_With_Volume
Concept: 6h Turtle Soup pattern with 1d trend filter and volume confirmation.
- Long when price makes new 20-bar low but closes above it (bull trap reversal)
- Short when price makes new 20-bar high but closes below it (bear trap reversal)
- Uses 1d EMA50 as trend filter: only long in uptrend, short in downtrend
- Requires volume > 1.5x average for confirmation
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: catches failed breakouts at key levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Turtle_Soup_With_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === 1d: EMA50 trend filter ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h: Price arrays ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 6h: 20-period high/low for traps ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h: Volume confirmation ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Get values
        ema50_val = ema50_1d_aligned[i]
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
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
        
        if position == 0:
            # Long trap: price makes new 20-bar low but closes above it (bull trap)
            bull_trap = low_val <= low_20_val and close_val > low_20_val
            # Short trap: price makes new 20-bar high but closes below it (bear trap)
            bear_trap = high_val >= high_20_val and close_val < high_20_val
            
            # Volume confirmation
            vol_confirm = vol_ratio_val > 1.5
            
            # Trend filter: only long in uptrend, short in downtrend
            if bull_trap and vol_confirm and close_val > ema50_val:
                signals[i] = 0.25
                position = 1
            elif bear_trap and vol_confirm and close_val < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price makes new 20-bar high (failed continuation)
            if high_val >= high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price makes new 20-bar low (failed continuation)
            if low_val <= low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals