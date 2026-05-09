#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Turtle_Soup_1d"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Turtle Soup patterns
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's high and low
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align to 6h
    prev_high_6h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_6h = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Simple volume filter: current 6h volume > 50-period average
    vol_series = pd.Series(prices['volume'].values)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    volume_filter = prices['volume'].values > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(prev_high_6h[i]) or np.isnan(prev_low_6h[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price makes new low (< prev low) then reverses above prev low
            if low[i] < prev_low_6h[i] and close[i] > prev_low_6h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price makes new high (> prev high) then reverses below prev high
            elif high[i] > prev_high_6h[i] and close[i] < prev_high_6h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below the low of the setup bar
            if low[i] < prev_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above the high of the setup bar
            if high[i] > prev_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals