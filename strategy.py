#!/usr/bin/env python3
"""
6h_WeeklyPivot_PriceAction_Scalp_v1
Concept: 6h price action around weekly pivot points with volume confirmation.
- Weekly pivot points calculated from prior week's OHLC
- Long: Price above weekly pivot + breaking above weekly R1 with volume surge
- Short: Price below weekly pivot + breaking below weekly S1 with volume surge
- Exit: Price returns to weekly pivot level
- Position sizing: 0.25
- Works in bull/bear: Uses weekly structure as dynamic support/resistance
- Target: 20-40 trades/year (80-160 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "6h_WeeklyPivot_PriceAction_Scalp_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    
    # Align weekly levels to 6h timeframe (no delay needed as levels are static for the week)
    pivot_64 = align_ltf_to_htf(prices, df_weekly, pivot)
    r1_64 = align_ltf_to_htf(prices, df_weekly, r1)
    s1_64 = align_ltf_to_htf(prices, df_weekly, s1)
    
    # Volume surge detection: current volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is unavailable
        if (np.isnan(pivot_64[i]) or np.isnan(r1_64[i]) or np.isnan(s1_64[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price above pivot, breaking above R1 with volume surge
            if (prices['close'][i] > pivot_64[i] and 
                prices['close'][i] > r1_64[i] and 
                prices['close'][i-1] <= r1_64[i-1] and 
                vol_surge):
                signals[i] = 0.25
                position = 1
            # Short: Price below pivot, breaking below S1 with volume surge
            elif (prices['close'][i] < pivot_64[i] and 
                  prices['close'][i] < s1_64[i] and 
                  prices['close'][i-1] >= s1_64[i-1] and 
                  vol_surge):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below pivot
            if prices['close'][i] <= pivot_64[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above pivot
            if prices['close'][i] >= pivot_64[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def align_ltf_to_htf(ltf_prices, htf_df, htf_values):
    """
    Helper function to align HTF values to LTF without look-ahead.
    Since weekly pivot levels are constant throughout the week,
    we simply forward-fill the values.
    """
    # Create a series with HTF values indexed by HTF timestamps
    htf_series = pd.Series(htf_values, index=htf_df.index)
    # Reindex to LTF timestamps, forward-filling to hold value until next weekly update
    aligned_series = htf_series.reindex(ltf_prices.index, method='ffill')
    return aligned_series.values