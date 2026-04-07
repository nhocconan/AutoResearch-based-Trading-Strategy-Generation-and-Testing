#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v7
Hypothesis: On 12-hour timeframe, use daily Camarilla pivot levels with volume confirmation and trend filter.
Go long when price touches or crosses above Camarilla H4 resistance with volume > 1.5x average.
Go short when price touches or crosses below Camarilla L4 support with volume > 1.5x average.
Exit when price reaches opposite H3/L3 level or reverses back through H4/L4.
Designed for 15-30 trades/year to minimize fee drag while capturing institutional reaction to key levels.
Works in both bull/bear markets as Camarilla adapts to volatility and volume filter confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v7"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    H4 = close_1d + 1.1/2 * (high_1d - low_1d)
    H3 = close_1d + 1.1/6 * (high_1d - low_1d)
    L3 = close_1d - 1.1/6 * (high_1d - low_1d)
    L4 = close_1d - 1.1/2 * (high_1d - low_1d)
    
    # Align daily levels to 12h timeframe (forward fill, shifted by 1 day)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(H4_12h[i]) or np.isnan(H3_12h[i]) or 
            np.isnan(L3_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H3 level or drops back below H4
            if close[i] >= H3_12h[i] or close[i] < H4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 level or rises back above L4
            if close[i] <= L3_12h[i] or close[i] > L4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price crosses above H4 level
                if close[i] > H4_12h[i] and close[i-1] <= H4_12h[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below L4 level
                elif close[i] < L4_12h[i] and close[i-1] >= L4_12h[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals