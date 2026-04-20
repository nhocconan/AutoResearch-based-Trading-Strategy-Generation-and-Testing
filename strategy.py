#!/usr/bin/env python3
"""
1d_Weekly_Alligator_Signal_v1
Concept: Weekly Williams Alligator (SMMA-based) defines trend direction, daily price closes above/below Alligator jaws trigger entries.
- Long: Weekly Alligator jaws (13-period SMMA) trending up AND daily close > jaws
- Short: Weekly Alligator jaws trending down AND daily close < jaws
- Exit: Price crosses back below/above jaws (trend change)
- Position sizing: 0.25
- Target: 15-30 trades/year (60-120 total over 4 years)
- Works in bull/bear: Weekly trend filter avoids counter-trend trades, Alligator adapts to volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Alligator_Signal_v1"
timeframe = "1d"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly: Williams Alligator (SMMA-based) ===
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # We only need jaws for trend direction
    weekly_close = df_1w['close'].values
    jaws = smma(weekly_close, 13)  # 13-period SMMA of weekly close
    jaws_aligned = align_htf_to_ltf(prices, df_1w, jaws)
    
    # === Daily: Price action ===
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for weekly jaws
    
    for i in range(start_idx, n):
        # Get values
        jaws_val = jaws_aligned[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(jaws_val) or np.isnan(close_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly jaws trending up AND daily close above jaws
            if i > start_idx:
                jaws_prev = jaws_aligned[i-1]
                jaws_trending_up = jaws_val > jaws_prev
            else:
                jaws_trending_up = False
            
            if jaws_trending_up and close_val > jaws_val:
                signals[i] = 0.25
                position = 1
            # Short: Weekly jaws trending down AND daily close below jaws
            elif i > start_idx:
                jaws_prev = jaws_aligned[i-1]
                jaws_trending_down = jaws_val < jaws_prev
            else:
                jaws_trending_down = False
                
            if jaws_trending_down and close_val < jaws_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Daily close crosses below jaws (trend change)
            if close_val < jaws_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Daily close crosses above jaws (trend change)
            if close_val > jaws_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals