#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Elder Ray (Bull/Bear Power) + 1w EMA50 trend filter.
Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
In strong uptrends (price > 1w EMA50), look for Bull Power expansion (>0 and rising) for longs.
In strong downtrends (price < 1w EMA50), look for Bear Power contraction (<0 and falling) for shorts.
Volume confirmation (>1.5x 20-period average) filters weak breakouts.
Position sizing: 0.25 for entries, 0 for exits.
Target: 50-150 total trades over 4 years (12-37/year).
Elder Ray captures institutional buying/selling pressure, effective in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_1d - ema13  # Bull Power = High - EMA13
    bear_power = low_1d - ema13   # Bear Power = Low - EMA13
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising (bullish expansion), volume spike, uptrend
            if (i > 0 and 
                bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling (bearish expansion), volume spike, downtrend
            elif (i > 0 and 
                  bear_power_aligned[i] < 0 and 
                  bear_power_aligned[i] < bear_power_aligned[i-1] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                  downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative or trend reverses
            if bull_power_aligned[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive or trend reverses
            if bear_power_aligned[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dElderRay_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0