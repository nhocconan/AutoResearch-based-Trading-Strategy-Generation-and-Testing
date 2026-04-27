#!/usr/bin/env python3
"""
#100787 - 6h_Range_Bound_1dMeanReversion_Volume
Hypothesis: Mean reversion from daily mean with volume confirmation on 6h timeframe.
Rationale: In sideways markets (common in 2025), price tends to revert to daily mean. Uses 1d EMA as dynamic mean with volume filter to avoid false signals. Works in both bull (pullbacks in uptrend) and bear (bounces in downtrend). Targets 15-30 trades/year to minimize fee drag.
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
    
    # Get 1d data for mean reversion target
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 as dynamic mean
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 6h timeframe (previous day's EMA for current period)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Distance from mean as percentage
    dist_from_mean = (close - ema20_1d_aligned) / ema20_1d_aligned * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price > 1.5% below mean with volume
        if (dist_from_mean[i] < -1.5 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price > 1.5% above mean with volume
        elif (dist_from_mean[i] > 1.5 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to within 0.5% of mean
        elif position == 1 and dist_from_mean[i] > -0.5:
            signals[i] = 0.0
            position = 0
        elif position == -1 and dist_from_mean[i] < 0.5:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Range_Bound_1dMeanReversion_Volume"
timeframe = "6h"
leverage = 1.0