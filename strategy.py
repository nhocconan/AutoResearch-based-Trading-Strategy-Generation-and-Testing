#!/usr/bin/env python3
"""
6h_1D_Pivot_R2S2_MomentumBreakout
Hypothesis: Trade breakouts at daily Camarilla R2/S2 levels on 6h timeframe with momentum confirmation.
Long when price breaks above R2 with upward momentum; short when breaks below S2 with downward momentum.
Uses 1d for pivot calculation and momentum filter to avoid false breakouts. Works in both bull and bear markets
as breakouts indicate momentum continuation regardless of broader trend.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "6h_1D_Pivot_R2S2_MomentumBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot calculation and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d momentum (close to close change) for trend filter
    mom_1d = df_1d['close'].pct_change().values
    mom_1d = np.where(np.isnan(mom_1d), 0, mom_1d)  # Replace NaN with 0 for first value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need at least 1 day of prior 6h data (4 bars) for pivot calc
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using prior day's OHLC from 1d data
        # Find the index of the prior completed day in 1d data
        # Each 6h bar = 0.25 days, so 4 bars = 1 day
        prior_day_idx = i // 4
        if prior_day_idx < 1:
            continue
            
        # Get prior day's OHLC from 1d data
        prior_day_high = df_1d['high'].iloc[prior_day_idx - 1]
        prior_day_low = df_1d['low'].iloc[prior_day_idx - 1]
        prior_day_close = df_1d['close'].iloc[prior_day_idx - 1]
        
        # Calculate Camarilla levels
        range_val = prior_day_high - prior_day_low
        if range_val <= 0:
            continue
            
        # Camarilla R2 and S2 levels
        r2 = prior_day_close + (range_val * 1.1 / 6)
        s2 = prior_day_close - (range_val * 1.1 / 6)
        
        current_close = prices['close'].iloc[i]
        
        # Momentum filter: require same direction as 1d momentum
        mom_direction = 1 if mom_1d[prior_day_idx - 1] > 0 else -1
        
        if position == 0:
            # Long: price breaks above R2 with upward momentum
            if current_close > r2 and mom_direction > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with downward momentum
            elif current_close < s2 and mom_direction < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S2 or momentum turns down
            if current_close < s2 or mom_1d[prior_day_idx - 1] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R2 or momentum turns up
            if current_close > r2 or mom_1d[prior_day_idx - 1] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals