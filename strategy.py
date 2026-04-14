# 6h_1dAlligator_1wPivot_Confluence_v1
# Hypothesis: 6h strategy using 1d Williams Alligator for trend direction and 1w pivot levels for support/resistance.
# The Alligator (three smoothed MAs) filters trend direction: price above all three = uptrend, below = downtrend.
# Weekly pivot levels provide key institutional support/resistance: long near S1/S2, short near R1/R2.
# Combines trend-following with mean-reversion at key levels to work in both bull and bear markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

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
    
    # Load 1d data ONCE for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Jaw: 13-period SMMA smoothed by 8
    # Teeth: 8-period SMMA smoothed by 5
    # Lips: 5-period SMMA smoothed by 3
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)  # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    # Align Alligator lines to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Load 1w data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using typical price)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Pivot, Support, Resistance levels
    pivot = tp_1w
    s1 = 2 * pivot - high_1w
    s2 = pivot - (high_1w - low_1w)
    s3 = low_1w + 2 * (pivot - high_1w)
    r1 = 2 * pivot - low_1w
    r2 = pivot + (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    
    # Align pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 34  # Need Alligator calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(r2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator trend filter
        # All lines aligned and ordered: Lips > Teeth > Jaw = uptrend
        # Lips < Teeth < Jaw = downtrend
        lips = lips_aligned[i]
        teeth = teeth_aligned[i]
        jaw = jaw_aligned[i]
        
        uptrend = lips > teeth and teeth > jaw
        downtrend = lips < teeth and teeth < jaw
        
        if position == 0:
            # Look for entries near weekly pivot levels with trend alignment
            price = close[i]
            pivot_val = pivot_aligned[i]
            s1_val = s1_aligned[i]
            s2_val = s2_aligned[i]
            r1_val = r1_aligned[i]
            r2_val = r2_aligned[i]
            
            # Long: price near support levels in uptrend
            near_support = (abs(price - s1_val) / s1_val < 0.01) or (abs(price - s2_val) / s2_val < 0.015)
            if uptrend and near_support:
                position = 1
                signals[i] = position_size
            # Short: price near resistance levels in downtrend
            elif downtrend and ((abs(price - r1_val) / r1_val < 0.01) or (abs(price - r2_val) / r2_val < 0.015)):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend reversal or price reaches pivot/resistance
            if not uptrend or close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend reversal or price reaches pivot/support
            if not downtrend or close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dAlligator_1wPivot_Confluence_v1"
timeframe = "6h"
leverage = 1.0