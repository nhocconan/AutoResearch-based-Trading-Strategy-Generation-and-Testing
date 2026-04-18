#!/usr/bin/env python3
"""
1d_Weekly_Range_Breakout
Hypothesis: Uses weekly price range (high-low) to identify expansion/contraction cycles.
Enters long when price breaks above weekly high with volume confirmation in low volatility environments,
and short when price breaks below weekly high in high volatility environments.
Designed to capture breakouts after consolidation in both bull and bear markets.
Target: 10-20 trades/year to minimize fee decay.
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
    volume = prices['volume'].values
    
    # Get weekly data for range and volatility
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly high, low, and range
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    range_1w = high_1w - low_1w
    
    # Calculate weekly ATR for volatility filter
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - df_1w['close'].values[:-1]),
            np.abs(low_1w[1:] - df_1w['close'].values[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full(len(tr_1w), np.nan)
    for i in range(14, len(tr_1w)):
        atr_1w[i] = np.nanmean(tr_1w[i-14:i])
    
    # Align weekly data to daily
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    range_1w_aligned = align_htf_to_ltf(prices, df_1w, range_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Volatility regime: low volatility when weekly ATR < 50-day average
    atr_ma = np.full(n, np.nan)
    for i in range(50, n):
        atr_ma[i] = np.mean(atr_1w_aligned[i-50:i])
    low_vol = atr_1w_aligned < atr_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or 
            np.isnan(range_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly high in low volatility
            if (close[i] > high_1w_aligned[i] and vol_confirm[i] and low_vol[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly low in high volatility
            elif (close[i] < low_1w_aligned[i] and vol_confirm[i] and not low_vol[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to weekly low or volatility increases
            if (close[i] < low_1w_aligned[i]) or (not low_vol[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly high or volatility decreases
            if (close[i] > high_1w_aligned[i]) or (low_vol[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Range_Breakout"
timeframe = "1d"
leverage = 1.0