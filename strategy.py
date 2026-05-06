#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams Alligator (SMMA) for trend direction and 1-week momentum for entry timing
# Long when price > Alligator's Jaw (13-period SMMA) and weekly ROC > 0, short when price < Jaw and weekly ROC < 0
# Williams Alligator uses smoothed moving averages (SMMA) with specific periods: Jaw(13), Teeth(8), Lips(5)
# Weekly ROC provides higher timeframe momentum confirmation to avoid counter-trend entries
# Designed to capture trends in both bull and bear markets by following the Alligator's alignment
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "4h_WilliamsAlligator_1wROC_Trend"
timeframe = "4h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(source) < length:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: (prev_smma * (length-1) + current_price) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need at least 13 for Jaw
        return np.zeros(n)
    
    # Calculate Williams Alligator components (SMMA)
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)    # Jaw: 13-period SMMA
    teeth = smma(close_1d, 8)   # Teeth: 8-period SMMA
    lips = smma(close_1d, 5)    # Lips: 5-period SMMA
    
    # Align Alligator lines to 4h timeframe (wait for daily bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1-week data for momentum confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need minimum for ROC
        return np.zeros(n)
    
    # Calculate weekly Rate of Change (ROC) - 5 period
    close_1w = df_1w['close'].values
    roc_1w = np.full_like(close_1w, np.nan, dtype=np.float64)
    for i in range(5, len(close_1w)):
        if close_1w[i-5] != 0:
            roc_1w[i] = ((close_1w[i] - close_1w[i-5]) / close_1w[i-5]) * 100
    
    # Align weekly ROC to 4h timeframe
    roc_1w_aligned = align_htf_to_ltf(prices, df_1w, roc_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(roc_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: price above Jaw AND teeth above lips (bullish alignment) AND weekly ROC positive
            if (close[i] > jaw_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and 
                roc_1w_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price below Jaw AND teeth below lips (bearish alignment) AND weekly ROC negative
            elif (close[i] < jaw_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and 
                  roc_1w_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Teeth (8-period SMMA) or weekly ROC turns negative
            if close[i] < teeth_aligned[i] or roc_1w_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Teeth (8-period SMMA) or weekly ROC turns positive
            if close[i] > teeth_aligned[i] or roc_1w_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#%%