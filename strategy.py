#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Trend_v1
Williams Alligator system: Jaw (13), Teeth (8), Lips (5) SMAs.
Long when Lips > Teeth > Jaw (bullish alignment), short when Lips < Teeth < Jaw (bearish).
Weekly trend filter: price above/below weekly EMA50.
Exit when Alligator alignment breaks.
Designed to catch trends with smoothed filters to reduce whipsaw.
Target: 20-50 total trades over 4 years (5-12/year).
"""

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
    
    # === Williams Alligator (using SMAs) ===
    # Jaw: 13-period SMMA (using SMA as approximation)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # === Weekly EMA50 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Bearish alignment: Lips < Teeth < Jaw
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: alignment breaks
        elif position == 1:
            # Exit long: bullish alignment broken (Lips <= Teeth or Teeth <= Jaw)
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bearish alignment broken (Lips >= Teeth or Teeth >= Jaw)
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_Trend_v1"
timeframe = "1d"
leverage = 1.0