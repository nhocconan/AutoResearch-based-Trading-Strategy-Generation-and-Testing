#!/usr/bin/env python3
# 6h_weekly_pivot_momentum_v1
# Hypothesis: Uses weekly pivot points from 1w data to establish long-term bias, combined with 6-hour momentum and volume confirmation.
# In bull markets (price above weekly pivot), look for long entries on 6h momentum breakouts with volume surge.
# In bear markets (price below weekly pivot), look for short entries on 6h momentum breakdowns with volume surge.
# Uses weekly support/resistance levels (S1, R1) for stop/reversal logic to adapt to both trending and ranging markets.
# Designed for low trade frequency (~20-50/year) to minimize fee drag on 6h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points and trend bias
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly data to 6h timeframe (using prior week's values for look-ahead protection)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # 6h momentum: 12-period ROC (rate of change)
    roc_period = 12
    roc = ((close / np.roll(close, roc_period)) - 1) * 100
    # Handle first roc_period values
    roc[:roc_period] = 0
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(roc[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Determine market bias from weekly pivot
        bullish_bias = close[i] > pivot_1w_aligned[i]
        bearish_bias = close[i] < pivot_1w_aligned[i]
        
        # Volume confirmation (2x average volume)
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below S1 or momentum reverses
            if close[i] < s1_1w_aligned[i] or roc[i] < -1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above R1 or momentum reverses
            if close[i] > r1_1w_aligned[i] or roc[i] > 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: bullish bias + positive momentum breakout
                if bullish_bias and roc[i] > 1.5 and close[i] > r1_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: bearish bias + negative momentum breakout
                elif bearish_bias and roc[i] < -1.5 and close[i] < s1_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals