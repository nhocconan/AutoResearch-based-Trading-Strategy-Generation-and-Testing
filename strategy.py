#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) + 1-week EMA50 filter
# Bull Power = High - EMA13 (daily), Bear Power = Low - EMA13 (daily)
# Long: Bull Power > 0 AND Bear Power < 0 AND weekly EMA50 rising
# Short: Bear Power < 0 AND Bull Power > 0 AND weekly EMA50 falling
# Exit: Bull/Bear Power crosses zero
# Elder Ray measures bull/bear strength relative to EMA13; weekly EMA50 filters trend.
# Target: 15-25 trades/year per symbol. Works in bull (strong bull power) and bear (strong bear power).
name = "6h_ElderRay_EMA13_WeeklyEMA50Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1-day data for EMA13 (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily data
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # 1-week data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D data to 6H timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Align 1W data to 6H timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        ema50 = ema50_1w_aligned[i]
        
        if position == 0:
            # Long entry: Bull Power > 0, Bear Power < 0, weekly EMA50 rising
            if (bull_power > 0 and bear_power < 0 and 
                ema50 > ema50_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0, Bull Power > 0, weekly EMA50 falling
            elif (bear_power < 0 and bull_power > 0 and 
                  ema50 < ema50_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power crosses below zero
            if bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power crosses above zero
            if bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals