#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R Regime Filter
# Combines Elder Ray (bull/bear power) for momentum strength with Williams %R for overbought/oversold extremes
# Uses 6h timeframe to balance responsiveness and noise reduction
# 1d Williams %R acts as regime filter: only take Elder Ray signals when not in extreme territory
# This avoids buying strength in overbought conditions or selling weakness in oversold conditions
# Works in both bull and bear markets by focusing on momentum extremes within non-extreme regimes
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Discrete position sizing: 0.25 (25% of capital) for balanced risk exposure

name = "6h_ElderRay_WilliamsR_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Measures bullish strength (high vs EMA)
    bear_power = low - ema13   # Measures bearish strength (low vs EMA)
    
    # Calculate 1d Williams %R for regime filter (14-period standard)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(13, 14)
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R regime: avoid extremes (> -20 = overbought, < -80 = oversold)
        # Only trade when -80 <= Williams %R <= -20 (non-extreme territory)
        in_regime = (williams_r_aligned[i] >= -80) and (williams_r_aligned[i] <= -20)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: strong bull power (buying conviction) AND in non-extreme regime
            if (bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > abs(bear_power_aligned[i]) and 
                in_regime):
                signals[i] = 0.25
                position = 1
            # Short entry: strong bear power (selling conviction) AND in non-extreme regime
            elif (bear_power_aligned[i] < 0 and 
                  abs(bear_power_aligned[i]) > bull_power_aligned[i] and 
                  in_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bear power exceeds bull power (momentum shift) OR Williams %R enters extreme
            if (bear_power_aligned[i] > bull_power_aligned[i]) or (williams_r_aligned[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bull power exceeds bear power (momentum shift) OR Williams %R enters extreme
            if (bull_power_aligned[i] > abs(bear_power_aligned[i])) or (williams_r_aligned[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals