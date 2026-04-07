#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: 12-hour Camarilla pivot levels act as strong support/resistance when combined with daily trend filter and volume confirmation.
Long when price breaks above H3 with daily uptrend and volume spike.
Short when price breaks below L3 with daily downtrend and volume spike.
Works in both bull and bear markets by following the daily trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3/L3 are key resistance/support
    h3_1d = close_1d + (range_1d * 1.1 / 4)
    l3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Daily trend filter: EMA25
    close_1d_series = pd.Series(close_1d)
    ema_25_1d = close_1d_series.ewm(span=25, min_periods=25).mean().values
    ema_25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_25_1d)
    
    # Volume confirmation: volume > 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema_25_1d_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below L3 or daily trend turns down
            if close[i] < l3_1d_aligned[i] or close[i] < ema_25_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above H3 or daily trend turns up
            if close[i] > h3_1d_aligned[i] or close[i] > ema_25_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above H3 with daily uptrend and volume confirmation
            if (close[i] > h3_1d_aligned[i] and 
                close[i] > ema_25_1d_aligned[i] and 
                volume[i] > vol_ma[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below L3 with daily downtrend and volume confirmation
            elif (close[i] < l3_1d_aligned[i] and 
                  close[i] < ema_25_1d_aligned[i] and 
                  volume[i] > vol_ma[i]):
                position = -1
                signals[i] = -0.25
    
    return signals