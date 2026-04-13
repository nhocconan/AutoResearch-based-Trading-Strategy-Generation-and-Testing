#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extremes with 1w Camarilla pivot continuation
    # Long: Williams %R < -80 (oversold) AND price > weekly R4 pivot (breakout)
    # Short: Williams %R > -20 (overbought) AND price < weekly S4 pivot (breakdown)
    # Exit: Williams %R returns to -50 (mean reversion) or opposite extreme
    # Using 6h for primary timeframe, Williams %R for momentum extremes,
    # Weekly Camarilla pivots for institutional breakout levels.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = Pivot + Range * 1.1/2
    # S4 = Pivot - Range * 1.1/2
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = pivot_1w + (range_1w * 1.1 / 2.0)
    s4_1w = pivot_1w - (range_1w * 1.1 / 2.0)
    
    # Align weekly Camarilla levels to 6h
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            highest_high[i] = np.max(high[i-lookback+1:i+1])
            lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, ((highest_high - close) / denominator) * -100, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (williams_r[i] < -80) and (close[i] > r4_1w_aligned[i])
        short_entry = (williams_r[i] > -20) and (close[i] < s4_1w_aligned[i])
        
        # Exit conditions: Williams %R returns to neutral zone
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_williamsr_camarilla_v1"
timeframe = "6h"
leverage = 1.0