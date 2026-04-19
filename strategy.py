#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Turtle Soup + daily ATR filter for mean reversion in range-bound markets.
# Long when price makes a new 20-bar low but closes above the low + 0.5*ATR(14) (false breakdown).
# Short when price makes a new 20-bar high but closes below the high - 0.5*ATR(14) (false breakout).
# Uses 1d ATR(14) as volatility filter to avoid low-volatility chop.
# Works in both bull and bear by capturing mean reversion after false breakouts.
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years).
name = "6h_TurtleSoup_DailyATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period high/low for Turtle Soup
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on daily timeframe
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 6h timeframe (wait for daily close)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20-period high/low data
    
    for i in range(start_idx, n):
        # Skip if ATR data is not available
        if np.isnan(atr_14_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_aligned[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        
        # Skip if ATR is too low (avoid choppy markets)
        if atr <= 0:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: false breakdown below 20-day low
            if low[i] < low_20_val and close[i] > low_20_val + 0.5 * atr:
                signals[i] = 0.25
                position = 1
            # Enter short: false breakout above 20-day high
            elif high[i] > high_20_val and close[i] < high_20_val - 0.5 * atr:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price reaches 20-day high or closes below entry area
            if high[i] >= high_20_val or close[i] < low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price reaches 20-day low or closes above entry area
            if low[i] <= low_20_val or close[i] > high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals