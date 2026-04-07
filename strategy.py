#!/usr/bin/env python3
"""
12h_camarilla_pivot_volume_filter_v1
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels from the previous daily session with volume confirmation to capture institutional-driven reversals at key support/resistance levels. Works in both bull and bear markets by fading extremes at pivot levels (L4/S4 for longs, H3/H4 for shorts) with volume filtering to avoid false breaks. Targets 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_filter_v1"
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
    
    # Get daily OHLC for Camarilla pivot calculation (use 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels
    H4 = close_1d + 1.5 * (high_1d - low_1d)
    H3 = close_1d + 1.0 * (high_1d - low_1d)
    L3 = close_1d - 1.0 * (high_1d - low_1d)
    L4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align to 12h timeframe (shifted by 1 day to avoid look-ahead)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume filter: 20-period average on 12h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (take profit) or breaks below L4 (stop)
            if close[i] >= H3_aligned[i] or close[i] <= L4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 (take profit) or breaks above H4 (stop)
            if close[i] <= L3_aligned[i] or close[i] >= H4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long entry: price touches or goes below L4 with rejection (close > L4)
                if close[i] <= L4_aligned[i] and close[i] > L4_aligned[i] * 0.999:  # touched L4
                    # Additional confirmation: price closing back above L4 (bullish rejection)
                    if i > 0 and close[i] > close[i-1]:
                        position = 1
                        signals[i] = 0.25
                # Short entry: price touches or goes above H4 with rejection (close < H4)
                elif close[i] >= H4_aligned[i] and close[i] < H4_aligned[i] * 1.001:  # touched H4
                    # Additional confirmation: price closing back below H4 (bearish rejection)
                    if i > 0 and close[i] < close[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals