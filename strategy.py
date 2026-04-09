#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v5"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for weekly high/low
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly high and low (using prior week's OHLC)
    week_high = np.full(len(df_w), np.nan)
    week_low = np.full(len(df_w), np.nan)
    for i in range(1, len(df_w)):
        week_high[i] = float(df_w['high'].iloc[i-1])
        week_low[i] = float(df_w['low'].iloc[i-1])
    
    # Align weekly values to daily timeframe
    week_high_aligned = align_htf_to_ltf(prices, df_w, week_high)
    week_low_aligned = align_htf_to_ltf(prices, df_w, week_low)
    
    # Volume confirmation: 3-day average
    vol_ma_3 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 3:
            vol_sum -= volume[i-3]
        if i >= 2:
            vol_ma_3[i] = vol_sum / 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(week_high_aligned[i]) or 
            np.isnan(week_low_aligned[i]) or 
            np.isnan(vol_ma_3[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly low
            if close[i] < week_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly high
            if close[i] > week_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above weekly high with volume confirmation
            vol_ratio = volume[i] / vol_ma_3[i] if vol_ma_3[i] > 0 else 0
            if close[i] > week_high_aligned[i] and vol_ratio > 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below weekly low with volume confirmation
            elif close[i] < week_low_aligned[i] and vol_ratio > 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals