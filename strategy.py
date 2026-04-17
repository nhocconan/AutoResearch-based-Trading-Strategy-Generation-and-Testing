#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Elder Ray + 1d Regime filter.
Long when Bull Power > 0 AND Bear Power < 0 AND price > 200 EMA (bull regime).
Short when Bear Power > 0 AND Bull Power < 0 AND price < 200 EMA (bear regime).
Elder Ray measures bull/bear power via EMA13; 200 EMA defines regime. Works in bull (trend continuation) and bear (mean reversion after volatility spikes).
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get daily data for Elder Ray and regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA13 for Elder Ray
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Calculate daily EMA200 for regime filter
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema200_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > EMA200 (bull regime)
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                close[i] > ema200_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND price < EMA200 (bear regime)
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < 0 and 
                  close[i] < ema200_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR price < EMA200 (regime change)
            if (bull_power_aligned[i] <= 0 or 
                bear_power_aligned[i] >= 0 or 
                close[i] < ema200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power <= 0 OR Bull Power >= 0 OR price > EMA200 (regime change)
            if (bear_power_aligned[i] <= 0 or 
                bull_power_aligned[i] >= 0 or 
                close[i] > ema200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dElderRay_Regime"
timeframe = "6h"
leverage = 1.0