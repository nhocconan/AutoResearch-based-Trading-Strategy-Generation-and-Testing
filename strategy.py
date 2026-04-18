#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_With_Volume_Confirmation
Hypothesis: Daily price breaks above/below weekly pivot resistance/support levels (R1/S1) with volume spike confirmation.
In bull markets, captures breakouts above R1; in bear markets, captures breakdowns below S1.
Weekly pivots act as institutional support/resistance levels. Volume confirms momentum behind the break.
Designed for 15-25 trades/year to minimize fee drag while capturing significant directional moves.
Works in both bull and bear markets by trading breakouts in direction of prevailing trend.
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
    
    # Weekly Pivot Points (calculated from weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w  # Resistance 1
    s1_1w = 2 * pivot_1w - high_1w  # Support 1
    
    # Align weekly pivots to daily timeframe (wait for weekly close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume spike: >2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if price > r1 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike
            elif price < s1 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to pivot level (mean reversion)
            if price <= pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to pivot level (mean reversion)
            if price >= pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0