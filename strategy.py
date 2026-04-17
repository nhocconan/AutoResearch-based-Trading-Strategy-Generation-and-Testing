#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA50 trend filter and ATR-based stops.
Long when Bull Power > 0, Bear Power < 0, and 12h EMA50 rising.
Short when Bear Power < 0, Bull Power > 0, and 12h EMA50 falling.
Exit when power signals reverse or price touches 12h EMA50.
Uses 6h for execution and Elder Ray calculation, 12h for EMA trend filter.
Elder Ray captures bull/bear power relative to EMA13, effective in both trending and ranging markets.
Target: 12-30 trades/year per symbol.
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = ema_50_12h > np.roll(ema_50_12h, 1)
    ema_50_falling = ema_50_12h < np.roll(ema_50_12h, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Align 12h EMA to 6h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # Calculate Elder Ray on 6h data
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, and 12h EMA50 rising
            if (bull_power[i] > 0 and bear_power[i] < 0 and ema_50_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, and 12h EMA50 falling
            elif (bear_power[i] < 0 and bull_power[i] > 0 and ema_50_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 or Bear Power >= 0 or 12h EMA50 falling
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or not ema_50_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 or Bull Power <= 0 or 12h EMA50 rising
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or not ema_50_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA50Trend"
timeframe = "6h"
leverage = 1.0