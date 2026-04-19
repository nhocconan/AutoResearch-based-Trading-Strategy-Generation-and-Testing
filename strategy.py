# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Pivot_HighLow_Breakout_Volume
Hypothesis: Price breaks above/below prior day high/low with volume confirmation indicate momentum.
Use 1-day pivot high/low as dynamic support/resistance. Volume > 1.5x 20-period average confirms breakout.
Works in bull (breakouts continue) and bear (breakdowns continue) via symmetric logic.
Targets 15-25 trades/year per symbol with disciplined entries.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_HighLow_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day high and low
    prior_high = df_1d['high'].shift(1).values  # shift(1) for completed day only
    prior_low = df_1d['low'].shift(1).values
    
    # Align to 6h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is valid
    
    for i in range(start_idx, n):
        # Skip if pivot levels not available (first day)
        if np.isnan(prior_high_aligned[i]) or np.isnan(prior_low_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above prior day high with volume
            if close[i] > prior_high_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below prior day low with volume
            elif close[i] < prior_low_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below prior day low
            if close[i] < prior_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above prior day high
            if close[i] > prior_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals