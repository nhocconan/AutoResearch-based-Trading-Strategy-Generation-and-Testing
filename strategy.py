#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v1
Hypothesis: Camarilla pivot levels from daily timeframe combined with volume confirmation on 4h chart.
In long: price touches S3/S4 level with above-average volume and closes above the level.
In short: price touches R3/R4 level with above-average volume and closes below the level.
Uses daily pivot levels for institutional support/resistance, volume for confirmation of institutional interest.
Designed for 20-40 trades/year on 4h timeframe with clear reversal logic that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_S3 = prev_close - (range_1d * 1.1 / 6)
    camarilla_S4 = prev_close - (range_1d * 1.1 / 4)
    camarilla_R3 = prev_close + (range_1d * 1.1 / 6)
    camarilla_R4 = prev_close + (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price relative to Camarilla levels (touch or penetrate)
        touch_S3_S4 = (low[i] <= S3_aligned[i]) or (low[i] <= S4_aligned[i])
        touch_R3_R4 = (high[i] >= R3_aligned[i]) or (high[i] >= R4_aligned[i])
        
        # Close confirmation: close beyond the level with body
        close_beyond_S3_S4 = (close[i] > S3_aligned[i]) or (close[i] > S4_aligned[i])
        close_beyond_R3_R4 = (close[i] < R3_aligned[i]) or (close[i] < R4_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below S4
            if close[i] < S4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R4
            if close[i] > R4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S3/S4 with volume confirmation and closes above
            if touch_S3_S4 and vol_confirmed and close_beyond_S3_S4:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3/R4 with volume confirmation and closes below
            elif touch_R3_R4 and vol_confirmed and close_beyond_R3_R4:
                position = -1
                signals[i] = -0.25
    
    return signals