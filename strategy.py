#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_Volume
Hypothesis: Weekly pivot points (R1, S1) act as strong weekly support/resistance.
Breakouts beyond these levels with volume confirmation capture momentum.
Designed for 1d timeframe with low trade frequency (<25/year) to minimize fee drag.
Works in bull/bear markets by requiring volume spike and using price action for direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using standard formula
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Use previous weekly bar's data to avoid look-ahead
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Shift by 1 to use previous weekly bar's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_prev)
    
    # Volume spike: 1.5x 20-period average on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike
            if price > r1_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike
            elif price < s1_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or closes below weekly pivot
            if price <= s1_val or price < pivot.iloc[i] if hasattr(pivot, 'iloc') else price < pivot[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or closes above weekly pivot
            if price >= r1_val or price > pivot.iloc[i] if hasattr(pivot, 'iloc') else price > pivot[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0