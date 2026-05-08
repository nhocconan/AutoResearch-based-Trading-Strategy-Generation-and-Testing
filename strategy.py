#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Pivot_Range_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support 1 = (2 * Pivot) - High
    s1_1w = (2 * pivot_1w) - high_1w
    # Resistance 1 = (2 * Pivot) - Low
    r1_1w = (2 * pivot_1w) - low_1w
    
    # Align weekly pivots to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate range for mean reversion
    range_1w = r1_1w - s1_1w
    range_1w_aligned = align_htf_to_ltf(prices, df_1w, range_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(len(prices)):
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(range_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near or below S1 with reversal signal
            if close[i] <= s1_1w_aligned[i] * 1.02 and close[i] >= s1_1w_aligned[i] * 0.98:
                # Look for bullish reversal: current close > open and higher than previous close
                if close[i] > prices['open'].iloc[i] and close[i] > close[i-1]:
                    signals[i] = 0.25
                    position = 1
            # Short: price near or above R1 with reversal signal
            elif close[i] >= r1_1w_aligned[i] * 0.98 and close[i] <= r1_1w_aligned[i] * 1.02:
                # Look for bearish reversal: current close < open and lower than previous close
                if close[i] < prices['open'].iloc[i] and close[i] < close[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches pivot or shows weakness
            if close[i] >= pivot_1w_aligned[i] * 0.99 or close[i] < close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches pivot or shows strength
            if close[i] <= pivot_1w_aligned[i] * 1.01 or close[i] > close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals