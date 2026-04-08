#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_v3
Hypothesis: Use 12h price action with 1d Camarilla pivot levels and volume confirmation for breakout trading.
Long when 12h price breaks above 1d H3 with volume confirmation.
Short when 12h price breaks below 1d L3 with volume confirmation.
Camarilla levels are more responsive to recent price action than standard pivots, adapting better to volatility.
Volume filter reduces false breakouts. Target: 20-35 trades/year per symbol (80-140 total over 4 years).
Works in bull markets by capturing breakouts and in bear markets by catching breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_alt, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v3"
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla equations
    range_1d = high_1d - low_1d
    close_prev = close_1d  # Using same day's close as it's the last completed bar
    
    # Camarilla levels for intraday trading
    h3 = close_prev + (range_1d * 1.1 / 4)
    l3 = close_prev - (range_1d * 1.1 / 4)
    h4 = close_prev + (range_1d * 1.1 / 2)
    l4 = close_prev - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: volume > 1.3x average of last 30 periods
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 1d L3 (stop and reverse)
            if close[i] < l3_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 1d H3 (stop and reverse)
            if close[i] > h3_aligned[i]:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 1d H3 with volume
            if close[i] > h3_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 1d L3 with volume
            elif close[i] < l3_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals