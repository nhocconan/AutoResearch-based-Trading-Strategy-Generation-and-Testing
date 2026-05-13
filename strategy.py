#!/usr/bin/env python3
name = "6h_RangeBreakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for range detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily range (high-low)
    daily_range = high_1d - low_1d
    # Calculate range percentile (20-period)
    range_series = pd.Series(daily_range)
    range_pct = range_series.rolling(window=20, min_periods=20).apply(
        lambda x: (x[-1] <= np.percentile(x, 30)) * 1.0, raw=True
    ).values
    
    # Align range filter to 6H
    range_pct_aligned = align_htf_to_ltf(prices, df_1d, range_pct)
    
    # Volume spike detection (6H)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(volume_ma > 0, volume_ma, 1)
    volume_spike = (volume_ratio > 1.5).astype(float)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if np.isnan(range_pct_aligned[i]) or np.isnan(volume_spike[i]):
            signals[i] = 0.0
            continue
        
        # Range-bound market (low volatility)
        in_range = range_pct_aligned[i] > 0.5
        # Volume spike indicates breakout attempt
        vol_spike = volume_spike[i] > 0
        
        if position == 0:
            # Look for breakout with volume
            if in_range and vol_spike:
                if close[i] > high[i-1]:  # Break above recent high
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low[i-1]:  # Break below recent low
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit on range expansion or reversal
            if not in_range or close[i] < close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if not in_range or close[i] > close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals