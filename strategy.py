#!/usr/bin/env python3
# 6h_1d_camarilla_reversal_v1
# Hypothesis: 6-hour Camarilla pivot reversal strategy using daily pivot levels.
# Go long when price touches S3 and closes back above it with volume confirmation.
# Go short when price touches R3 and closes back below it with volume confirmation.
# Uses 1d Camarilla levels calculated from previous day's OHLC.
# Works in ranging markets (reversions at S3/R3) and trending markets (breakouts at S4/R4).
# Volume filter ensures participation, reducing false touches.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla multipliers
    S3 = prev_close - (prev_range * 1.1 / 6)
    S4 = prev_close - (prev_range * 1.1 / 2)
    R3 = prev_close + (prev_range * 1.1 / 6)
    R4 = prev_close + (prev_range * 1.1 / 2)
    
    # Align daily levels to 6h timeframe (using previous day's levels)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(S3_6h[i]) or np.isnan(R3_6h[i]) or 
            np.isnan(S4_6h[i]) or np.isnan(R4_6h[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S4 (strong support break) or returns to S3
            if close[i] <= S4_6h[i] or close[i] >= S3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 (strong resistance break) or returns to R3
            if close[i] >= R4_6h[i] or close[i] <= R3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches S3 and closes back above it with volume
            if (low[i] <= S3_6h[i] and close[i] > S3_6h[i] and 
                volume[i] > vol_ma_24[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R3 and closes back below it with volume
            elif (high[i] >= R3_6h[i] and close[i] < R3_6h[i] and 
                  volume[i] > vol_ma_24[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals