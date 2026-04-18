#!/usr/bin/env python3
"""
1d_Williams_Alligator_Trend_Filter_with_Volume_Confirmation
Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) defines trend direction (Lips above Teeth/Jaw = uptrend, below = downtrend).
Entry occurs on weekly Donchian(20) breakout in trend direction with volume spike.
Williams Alligator uses SMAs with specific offsets (Jaw=13*8, Teeth=8*5, Lips=5*3) to avoid whipsaw.
Volume surge confirms momentum. Designed for 10-25 trades/year to minimize fee decay while capturing strong trends.
Works in bull (catch breakouts) and bear (catch breakdowns) via symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 1d: Jaw(13*8), Teeth(8*5), Lips(5*3)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Jaw: Blue line, 13-period SMMA shifted 8 bars
    jaw_raw = pd.Series(close_1d).rolling(window=13*8, min_periods=13*8).mean().values
    jaw = np.roll(jaw_raw, 8*8)  # shift right by 8 bars
    jaw[:8*8] = np.nan
    
    # Teeth: Red line, 8-period SMMA shifted 5 bars
    teeth_raw = pd.Series(close_1d).rolling(window=8*5, min_periods=8*5).mean().values
    teeth = np.roll(teeth_raw, 5*5)  # shift right by 5 bars
    teeth[:5*5] = np.nan
    
    # Lips: Green line, 5-period SMMA shifted 3 bars
    lips_raw = pd.Series(close_1d).rolling(window=5*3, min_periods=5*3).mean().values
    lips = np.roll(lips_raw, 3*3)  # shift right by 3 bars
    lips[:3*3] = np.nan
    
    # Align Alligator lines to 1d index
    jaw_1d = jaw
    teeth_1d = teeth
    lips_1d = lips
    
    # Weekly Donchian(20) channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(13*8 + 8*8, 8*5 + 5*5, 5*3 + 3*3, 20)  # Warmup for Alligator and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        dc_high = donch_high_aligned[i]
        dc_low = donch_low_aligned[i]
        vol_spike = volume_spike[i]
        
        # Trend conditions: Lips above Teeth/Jaw = uptrend, below = downtrend
        uptrend = lips_val > teeth_val and lips_val > jaw_val
        downtrend = lips_val < teeth_val and lips_val < jaw_val
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume spike and uptrend
            if price > dc_high and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume spike and downtrend
            elif price < dc_low and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: trend changes to downtrend OR price touches weekly Donchian low
            if not uptrend or price <= dc_low:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: trend changes to uptrend OR price touches weekly Donchian high
            if not downtrend or price >= dc_high:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Williams_Alligator_Trend_Filter_with_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0