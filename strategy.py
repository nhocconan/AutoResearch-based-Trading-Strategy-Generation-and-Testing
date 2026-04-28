#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_R1S1_R4S4_Breakout_VolumeSpike
Hypothesis: Combines tight intraday R1/S1 breakouts (mean reversion in range) with 
breakout confirmation at weekly R4/S4 levels (trend continuation). Volume spike 
filters false signals. Works in ranging markets (fade at R1/S1) and trending 
markets (breakout at R4/S4). Targets 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 12:
        return np.zeros(n)
    
    # Daily data for R1/S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Weekly data for R4/S4
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily Camarilla R1/S1
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + hl_range_1d * 1.1 / 12.0
    s1_1d = close_1d - hl_range_1d * 1.1 / 12.0
    
    # Calculate weekly Camarilla R4/S4
    hl_range_1w = high_1w - low_1w
    r4_1w = close_1w + hl_range_1w * 1.1 / 2.0  # R4 = C + (H-L)*1.1/2
    s4_1w = close_1w - hl_range_1w * 1.1 / 2.0  # S4 = C - (H-L)*1.1/2
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Daily volume spike (>2x 20-period MA)
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Price levels
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        vol_spike = vol_spike_aligned[i] > 0.5
        
        # Midpoint for exits (daily close)
        midpoint_1d = close_1d
        midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
        midpoint_val = midpoint_aligned[i]
        
        # Entry logic:
        # Long: Price > R1 AND < R4 with volume spike (fade at R1, but not beyond R4)
        # Short: Price < S1 AND > S4 with volume spike (fade at S1, but not beyond S4)
        # Breakout continuation: Price > R4 OR < S4 with volume spike (trend continuation)
        long_fade = (close[i] > r1) and (close[i] < r4) and vol_spike
        short_fade = (close[i] < s1) and (close[i] > s4) and vol_spike
        long_breakout = (close[i] > r4) and vol_spike
        short_breakout = (close[i] < s4) and vol_spike
        
        # Exit: price returns to daily midpoint
        long_exit = close[i] < midpoint_val
        short_exit = close[i] > midpoint_val
        
        if (long_fade or long_breakout) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_fade or short_breakout) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_Pivot_R1S1_R4S4_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0