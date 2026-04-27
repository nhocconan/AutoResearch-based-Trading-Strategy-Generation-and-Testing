#!/usr/bin/env python3
"""
12h_Alligator_Trend_Filter_SmallSignal
Hypothesis: Williams Alligator with strict thresholds and small position size reduces whipsaw in chop.
Uses 1d Alligator (JAWS/TEETH/LIPS) for trend direction and 12h price action for entry.
Small position (0.15) and strict trend filter aim for <30 trades/year to avoid fee drag.
Works in bull by catching trends, in bear by avoiding false signals via strict Alligator alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    close_median = median_price.values
    
    # JAWS (13-period, 8-bar shift), TEETH (8-period, 5-bar shift), LIPS (5-period, 3-bar shift)
    jaws = pd.Series(close_median).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_median).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_median).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.15   # Small position size to reduce drawdown and trade frequency
    
    # Warmup: need enough data for Alligator (max shift 8)
    start_idx = 13 + 8  # 21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            signals[i] = 0.0
            continue
        
        jaws_val = jaws_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Trend conditions: Alligator alignment
        bullish = lips_val > teeth_val > jaws_val  # Green alignment
        bearish = lips_val < teeth_val < jaws_val  # Red alignment
        
        if position == 0:
            # Enter long only on strong bullish alignment with price above LIPS
            if bullish and close[i] > lips_val:
                signals[i] = size
                position = 1
            # Enter short only on strong bearish alignment with price below LIPS
            elif bearish and close[i] < lips_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: bearish alignment or price crosses below TEETH
            if bearish or close[i] < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish alignment or price crosses above TEETH
            if bullish or close[i] > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Alligator_Trend_Filter_SmallSignal"
timeframe = "12h"
leverage = 1.0