#!/usr/bin/env python3
"""
12h_1d_camarilla_volume_v1
Hypothesis: On 12h timeframe, price touching Camarilla pivot levels (H4/L4) from the previous 1-day with volume expansion captures reversals. The Camarilla levels provide strong support/resistance in ranging markets, which is beneficial during the 2025-2026 bear/range market. Volume confirmation filters false touches. Works in both bull and bear markets as it fades extremes at statistically significant levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day (using prior day's OHLC)
    # Camarilla: H4 = close + 1.5 * (high - low), L4 = close - 1.5 * (high - low)
    # Use previous day's data to avoid look-ahead
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 12h (using previous day's levels for current day)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume filter: 12h volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price touches H4 level or reaches opposite L4 level (mean reversion complete)
            if (close[i] >= h4_aligned[i]) or (close[i] <= l4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price touches L4 level or reaches opposite H4 level (mean reversion complete)
            if (close[i] <= l4_aligned[i]) or (close[i] >= h4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price touches L4 level with volume (support bounce)
            if (abs(close[i] - l4_aligned[i]) < 0.001 * close[i]) and volume_filter[i]:
                # More precisely: price within 0.1% of L4 level
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches H4 level with volume (resistance rejection)
            elif (abs(close[i] - h4_aligned[i]) < 0.001 * close[i]) and volume_filter[i]:
                # More precisely: price within 0.1% of H4 level
                position = -1
                signals[i] = -0.25
    
    return signals