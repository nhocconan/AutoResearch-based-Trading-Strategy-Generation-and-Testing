#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (based on previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    
    # Align weekly pivot levels to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate daily ATR for volatility filter
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for enough data
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price crosses above S1 (support) with rejection
            if close[i-1] < s1_1w_aligned[i] and close[i] > s1_1w_aligned[i]:
                # Additional confirmation: price stays above S1 for 2 consecutive closes
                if i >= 2 and close[i-1] > s1_1w_aligned[i-1] and close[i-2] > s1_1w_aligned[i-2]:
                    signals[i] = 0.25
                    position = 1
            # Short: price crosses below R1 (resistance) with rejection
            elif close[i-1] > r1_1w_aligned[i] and close[i] < r1_1w_aligned[i]:
                # Additional confirmation: price stays below R1 for 2 consecutive closes
                if i >= 2 and close[i-1] < r1_1w_aligned[i-1] and close[i-2] < r1_1w_aligned[i-2]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price reaches R1 or closes below S1
            if price >= r1_1w_aligned[i] or close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches S1 or closes above R1
            if price <= s1_1w_aligned[i] or close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals