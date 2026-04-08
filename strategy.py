#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_volume_v1
# Hypothesis: 4h price breaking above/below daily Camarilla pivot levels (H4/L4) with volume confirmation creates high-probability breakout trades.
# Uses 1d timeframe for Camarilla pivot calculation (proven support/resistance levels) and 4h for entry timing.
# Works in both bull/bear markets by trading breakouts in direction of prevailing trend.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
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
    open_prices = prices['open'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    h4 = close_1d + (range_1d * 1.1 / 2)  # Resistance 4
    l4 = close_1d - (range_1d * 1.1 / 2)  # Support 4
    h3 = close_1d + (range_1d * 1.1 / 4)  # Resistance 3
    l3 = close_1d - (range_1d * 1.1 / 4)  # Support 3
    
    # Align Camarilla levels to 4h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: volume > 1.3x average of last 20 periods (5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or \
           np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3
            if close[i] < l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above H3
            if close[i] > h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above H4 with volume
            if (close[i] > h4_1d_aligned[i] and 
                open_prices[i] <= h4_1d_aligned[i] and  # Ensure breakout happened this bar
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L4 with volume
            elif (close[i] < l4_1d_aligned[i] and 
                  open_prices[i] >= l4_1d_aligned[i] and  # Ensure breakdown happened this bar
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals