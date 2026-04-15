#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h weekly pivot breakout + volume confirmation
# Trade breakout above weekly pivot resistance (R1) or below support (S1) with volume > 1.5x median.
# Use weekly pivot levels from prior week for bias: price above weekly pivot = bullish bias (long breakouts only),
# price below weekly pivot = bearish bias (short breakouts only). This adapts to weekly trend.
# Target: 50-150 total trades over 4 years with discrete sizing (0.25) to limit fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly pivot points (calculated from prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Resistance levels: R1 = 2*P - L, R2 = P + (H-L)
    r1_1w = 2 * pivot_1w - low_1w
    # Support levels: S1 = 2*P - H, S2 = P - (H-L)
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly levels to 6h timeframe (using prior week's values)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long breakout: price > R1 weekly + volume spike + price above weekly pivot (bullish bias)
        if (close[i] > r1_1w_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            close[i] > pivot_1w_aligned[i]):
            signals[i] = 0.25
        
        # Short breakout: price < S1 weekly + volume spike + price below weekly pivot (bearish bias)
        elif (close[i] < s1_1w_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              close[i] < pivot_1w_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to weekly pivot area (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] <= pivot_1w_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] >= pivot_1w_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0