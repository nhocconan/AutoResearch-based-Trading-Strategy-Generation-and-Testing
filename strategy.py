#!/usr/bin/env python3
# 12h_camarilla_1d_volume_v1
# Hypothesis: 12h Camarilla pivot levels from 1d HTF + volume confirmation. 
# Camarilla levels act as institutional support/resistance where price often reacts.
# Volume confirms institutional participation at these levels. Works in both bull/bear markets
# as it fades extreme moves rather than following trends, avoiding whipsaw in ranging conditions.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Camarilla pivot levels
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    h4 = pivot_point + (range_1d * 1.1 / 2)
    h3 = pivot_point + (range_1d * 1.1 / 4)
    h2 = pivot_point + (range_1d * 1.1 / 6)
    h1 = pivot_point + (range_1d * 1.1 / 12)
    l1 = pivot_point - (range_1d * 1.1 / 12)
    l2 = pivot_point - (range_1d * 1.1 / 6)
    l3 = pivot_point - (range_1d * 1.1 / 4)
    l4 = pivot_point - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back below H3 (mean reversion)
            if close[i] < h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back above L3 (mean reversion)
            if close[i] > l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Fade extreme moves: short at H3, long at L3
                if close[i] > h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                elif close[i] < l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
    
    return signals