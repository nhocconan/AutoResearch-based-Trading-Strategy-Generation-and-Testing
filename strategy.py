#!/usr/bin/env python3
# 12h_1d_alligator_momentum_v1
# Hypothesis: 12-hour Williams Alligator with 1-day trend filter.
# Uses Alligator (Jaw, Teeth, Lips) on 12h for momentum and 1-day close > SMA50 for trend filter.
# Long when Lips > Teeth > Jaw (bullish alignment) and price above 1-day SMA50.
# Short when Lips < Teeth < Jaw (bearish alignment) and price below 1-day SMA50.
# Exit when Alligator alignment breaks or price crosses 1-day SMA50 in opposite direction.
# Designed to generate ~20-40 trades/year to avoid fee decay while capturing sustained trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_alligator_momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator parameters (13, 8, 5) with future shifts (8, 5, 3)
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw (Blue line): 13-period SMA of median, shifted 8 bars ahead
    jaw = np.full(n, np.nan)
    for i in range(jaw_period - 1, n):
        jaw[i] = np.mean(median_price[i - jaw_period + 1:i + 1])
    # Shift jaw forward by jaw_shift bars
    jaw_shifted = np.full(n, np.nan)
    for i in range(n - jaw_shift):
        jaw_shifted[i + jaw_shift] = jaw[i]
    
    # Teeth (Red line): 8-period SMA of median, shifted 5 bars ahead
    teeth = np.full(n, np.nan)
    for i in range(teeth_period - 1, n):
        teeth[i] = np.mean(median_price[i - teeth_period + 1:i + 1])
    # Shift teeth forward by teeth_shift bars
    teeth_shifted = np.full(n, np.nan)
    for i in range(n - teeth_shift):
        teeth_shifted[i + teeth_shift] = teeth[i]
    
    # Lips (Green line): 5-period SMA of median, shifted 3 bars ahead
    lips = np.full(n, np.nan)
    for i in range(lips_period - 1, n):
        lips[i] = np.mean(median_price[i - lips_period + 1:i + 1])
    # Shift lips forward by lips_shift bars
    lips_shifted = np.full(n, np.nan)
    for i in range(n - lips_shift):
        lips_shifted[i + lips_shift] = lips[i]
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1-day SMA50 for trend filter
    sma50_1d = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):  # 50-period SMA
        sma50_1d[i] = np.mean(close_1d[i - 49:i + 1])
    
    # Align 1-day SMA50 to 12h timeframe
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        sma50_1d_val = sma50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) or price below 1-day SMA50
            if lips_val <= teeth_val or teeth_val <= jaw_val or price < sma50_1d_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) or price above 1-day SMA50
            if lips_val >= teeth_val or teeth_val >= jaw_val or price > sma50_1d_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Alligator alignment with trend filter
            # Bullish: Lips > Teeth > Jaw (green > red > blue) and price above 1-day SMA50
            if lips_val > teeth_val and teeth_val > jaw_val and price > sma50_1d_val:
                position = 1
                signals[i] = 0.25
            # Bearish: Lips < Teeth < Jaw (green < red < blue) and price below 1-day SMA50
            elif lips_val < teeth_val and teeth_val < jaw_val and price < sma50_1d_val:
                position = -1
                signals[i] = -0.25
    
    return signals