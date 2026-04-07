#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v2
Hypothesis: Camarilla pivot levels from 1-day timeframe provide strong support/resistance.
Price reverting to these levels with volume confirmation offers high-probability mean-reversion trades.
Works in both bull and bear markets as it captures mean reversion within larger trends.
Uses 12h timeframe to limit trades and reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla equations
    # Resistance levels
    R4 = prev_close + ((prev_high - prev_low) * 1.5000)
    R3 = prev_close + ((prev_high - prev_low) * 1.2500)
    R2 = prev_close + ((prev_high - prev_low) * 1.1666)
    R1 = prev_close + ((prev_high - prev_low) * 1.0833)
    # Support levels
    S1 = prev_close - ((prev_high - prev_low) * 1.0833)
    S2 = prev_close - ((prev_high - prev_low) * 1.1666)
    S3 = prev_close - ((prev_high - prev_low) * 1.2500)
    S4 = prev_close - ((prev_high - prev_low) * 1.5000)
    
    # Align all levels to 12h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: 24-period average on 12h (2 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(vol_ma[i]) or
            vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or volume fails
            if close[i] >= R3_aligned[i] or not vol_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 or volume fails
            if close[i] <= S3_aligned[i] or not vol_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S1 with volume confirmation
            if close[i] <= S1_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: price touches R1 with volume confirmation
            elif close[i] >= R1_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals