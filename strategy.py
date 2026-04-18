#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeBreakout
6h strategy using weekly pivot points (S2/R2) with volatility filter.
- Long: Close breaks above weekly R2 + volatility expansion (ATR > 1.2x ATR(20))
- Short: Close breaks below weekly S2 + volatility expansion
- Exit: Opposite breakout or volatility contraction
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in trending markets (breakout continuation) and range markets (mean reversion at extremes)
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
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and filters
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using weekly high/low/close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R2 = pivot + (H - L)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # Weekly S2 = pivot - (H - L)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Weekly ATR for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all weekly data to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 6h ATR for volatility expansion filter
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for ATR20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion filter: current ATR > 1.2x weekly ATR
        vol_expansion = atr_6h[i] > 1.2 * atr_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r2_aligned[i]
        breakdown_down = close[i] < s2_aligned[i]
        
        if position == 0:
            # Long: volatility expansion + breakout above weekly R2
            if vol_expansion and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: volatility expansion + breakdown below weekly S2
            elif vol_expansion and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: volatility contraction or breakdown below weekly S2
            if not vol_expansion or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: volatility contraction or breakout above weekly R2
            if not vol_expansion or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_RangeBreakout"
timeframe = "6h"
leverage = 1.0