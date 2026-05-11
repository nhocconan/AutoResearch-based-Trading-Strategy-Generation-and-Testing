#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Trend_Follow"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points using previous week's OHLC
    # We use the previous week's data to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Resistance 1 = (2 * P) - L
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    # Support 1 = (2 * P) - H
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly trend filter: price above/below pivot
    weekly_trend_up = close > pivot_aligned
    weekly_trend_down = close < pivot_aligned
    
    # 6-period EMA for entry timing on 6h chart
    ema6_6h = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 6  # Ensure EMA is ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema6_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above weekly pivot, above S1, and EMA rising
            if (close[i] > pivot_aligned[i] and 
                close[i] > s1_aligned[i] and
                ema6_6h[i] > ema6_6h[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly pivot, below R1, and EMA falling
            elif (close[i] < pivot_aligned[i] and 
                  close[i] < r1_aligned[i] and
                  ema6_6h[i] < ema6_6h[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly pivot or S1
            if (close[i] < pivot_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly pivot or R1
            if (close[i] > pivot_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals