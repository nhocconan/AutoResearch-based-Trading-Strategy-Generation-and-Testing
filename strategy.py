#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirmation_v2
# Hypothesis: Camarilla pivot levels from daily charts act as strong support/resistance.
# Price breaking above R1 with volume confirmation indicates bullish momentum;
# breaking below S1 indicates bearish momentum. Filtered by 1-week EMA200 to avoid
# counter-trend trades in strong weekly trends. Volume confirmation requires
# current 4h volume > 1.5x 20-period average. Target: 20-50 trades/year (80-200 total).

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirmation_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Typical price
    typical_prev = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    R1 = typical_prev + (range_prev * 1.1 / 12)
    S1 = typical_prev - (range_prev * 1.1 / 12)
    
    # Align to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need previous day data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1-week EMA200
            uptrend = close[i] > ema200_1w_aligned[i]
            downtrend = close[i] < ema200_1w_aligned[i]
            
            # Long: uptrend + price > R1 + volume confirmation
            if uptrend and close[i] > R1_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price < S1 + volume confirmation
            elif downtrend and close[i] < S1_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or trend reverses
            if close[i] < S1_aligned[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or trend reverses
            if close[i] > R1_aligned[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals