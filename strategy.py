#!/usr/bin/env python3
"""
4h_Stochastic_Bollinger_Reversal
Hypothesis: Mean reversion at Bollinger Bands with Stochastic oversold/overbought conditions.
Long when price touches lower BB and Stochastic %K < 20 (oversold).
Short when price touches upper BB and Stochastic %K > 80 (overbought).
Filters: Bollinger Band width must be > 10th percentile (not in squeeze).
Timeframe: 4h to avoid overtrading; works in both bull/bear via mean reversion logic.
Position size: 0.25 to limit drawdown. Target: 20-40 trades/year.
"""

name = "4h_Stochastic_Bollinger_Reversal"
timeframe = "4h"
leverage = 1.0

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
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    basis = np.full_like(close, np.nan)
    dev = np.full_like(close, np.nan)
    upper = np.full_like(close, np.nan)
    lower = np.full_like(close, np.nan)
    
    for i in range(bb_length - 1, n):
        basis[i] = np.mean(close[i - bb_length + 1:i + 1])
        dev[i] = bb_mult * np.std(close[i - bb_length + 1:i + 1])
        upper[i] = basis[i] + dev[i]
        lower[i] = basis[i] - dev[i]
    
    # Stochastic Oscillator (14, 3, 3)
    stoch_k = 14
    stoch_smooth_k = 3
    stoch_smooth_d = 3
    
    lowest_low = np.full_like(low, np.nan)
    highest_high = np.full_like(high, np.nan)
    
    for i in range(stoch_k - 1, n):
        lowest_low[i] = np.min(low[i - stoch_k + 1:i + 1])
        highest_high[i] = np.max(high[i - stoch_k + 1:i + 1])
    
    stoch_raw = np.full_like(close, np.nan)
    for i in range(stoch_k - 1, n):
        if highest_high[i] != lowest_low[i]:
            stoch_raw[i] = (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i]) * 100
        else:
            stoch_raw[i] = 50.0
    
    stoch_k_smoothed = np.full_like(close, np.nan)
    for i in range(stoch_k - 1 + stoch_smooth_k - 1, n):
        start = i - stoch_smooth_k + 1
        end = i + 1
        stoch_k_smoothed[i] = np.mean(stoch_raw[start:end])
    
    stoch_d = np.full_like(close, np.nan)
    for i in range(stoch_k - 1 + stoch_smooth_k - 1 + stoch_smooth_d - 1, n):
        start = i - stoch_smooth_d + 1
        end = i + 1
        stoch_d[i] = np.mean(stoch_k_smoothed[start:end])
    
    # Bollinger Band Width % (for squeeze filter)
    bb_width = np.full_like(close, np.nan)
    for i in range(bb_length - 1, n):
        if basis[i] != 0:
            bb_width[i] = (upper[i] - lower[i]) / basis[i] * 100
        else:
            bb_width[i] = 0
    
    # BB width percentile (10-period lookback for threshold)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(9, len(bb_width)):  # need 10 values for percentile
        window = bb_width[i-9:i+1]
        if not np.all(np.isnan(window)):
            sorted_window = np.sort(window[~np.isnan(window)])
            idx = int(0.1 * len(sorted_window))  # 10th percentile
            bb_width_percentile[i] = sorted_window[idx] if len(sorted_window) > 0 else 0
        else:
            bb_width_percentile[i] = 0
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_length - 1, stoch_k - 1 + stoch_smooth_k - 1 + stoch_smooth_d - 1, 9) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(stoch_k_smoothed[i]) or 
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Only trade if not in Bollinger squeeze (width > 10th percentile)
        if bb_width[i] <= bb_width_percentile[i]:
            # In squeeze: stay flat or hold current position
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        if position == 0:
            # Long: price at or below lower BB AND Stochastic oversold (< 20)
            if close[i] <= lower[i] and stoch_k_smoothed[i] < 20:
                signals[i] = 0.25
                position = 1
            # Short: price at or above upper BB AND Stochastic overbought (> 80)
            elif close[i] >= upper[i] and stoch_k_smoothed[i] > 80:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses above midpoint OR Stochastic > 80 (overbought)
            midpoint = (upper[i] + lower[i]) / 2
            if close[i] > midpoint or stoch_k_smoothed[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below midpoint OR Stochastic < 20 (oversold)
            midpoint = (upper[i] + lower[i]) / 2
            if close[i] < midpoint or stoch_k_smoothed[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals