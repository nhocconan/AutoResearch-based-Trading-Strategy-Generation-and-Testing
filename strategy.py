#!/usr/bin/env python3
# Hypothesis: 1d timeframe with 1-week RSI extremes and weekly volume confirmation. 
# In overbought conditions (weekly RSI > 70) with high volume, price tends to reverse downward in both bull and bear markets. 
# In oversold conditions (weekly RSI < 30) with high volume, price tends to reverse upward. 
# Uses weekly RSI with volume filter to avoid false signals in low-volume environments. 
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_WeeklyRSI_Volume_Reversal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI (14-period)
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    
    # Calculate weekly volume average (20-period)
    vol_1w = df_1w['volume']
    vol_avg_1w = vol_1w.rolling(window=20, min_periods=20).mean()
    vol_avg_1w_values = vol_avg_1w.values
    
    # Align weekly indicators to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w_values)
    
    # Volume condition: current weekly volume > 1.5x average weekly volume
    # Need to get current weekly volume - use the weekly volume data aligned
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w.values)
    volume_condition = vol_1w_aligned > 1.5 * vol_avg_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(volume_condition[i]) or
            np.isnan(vol_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold (RSI < 30) + high volume
            if rsi_1w_aligned[i] < 30 and volume_condition[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought (RSI > 70) + high volume
            elif rsi_1w_aligned[i] > 70 and volume_condition[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or overbought
            if rsi_1w_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or oversold
            if rsi_1w_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals