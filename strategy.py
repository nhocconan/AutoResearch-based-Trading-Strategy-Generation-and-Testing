#!/usr/bin/env python3
"""
4h_camarilla_pivot_12h_volume_v1
Hypothesis: On 4-hour timeframe, use Camarilla pivot levels from 12-hour timeframe for support/resistance levels.
Enter long when price touches S1 level with volume confirmation (volume > 1.5x 20-period average).
Enter short when price touches R1 level with volume confirmation.
Exit when price reaches opposite pivot level (S3/R3) or reverses from touch point.
Camarilla levels provide statistically significant support/resistance; volume confirms institutional interest.
12h timeframe reduces noise vs 1d while capturing meaningful structure. Target: 25-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_12h_volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar
    # Based on previous day's high, low, close
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Pivot point
    pivot = (h_12h + l_12h + c_12h) / 3
    # Camarilla levels
    s1 = c_12h - (h_12h - l_12h) * 1.1 / 12
    s2 = c_12h - (h_12h - l_12h) * 1.1 / 6
    s3 = c_12h - (h_12h - l_12h) * 1.1 / 4
    r1 = c_12h + (h_12h - l_12h) * 1.1 / 12
    r2 = c_12h + (h_12h - l_12h) * 1.1 / 6
    r3 = c_12h + (h_12h - l_12h) * 1.1 / 4
    
    # Align all levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if pivot data not available
        if np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Price levels
        s1_level = s1_aligned[i]
        s3_level = s3_aligned[i]
        r1_level = r1_aligned[i]
        r3_level = r3_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price reaches S3 (strong support) or reverses from S1
            if close[i] <= s3_level:
                exit_long = True
            elif close[i] < s1_level and i > 1 and close[i-1] >= s1_level:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price reaches R3 (strong resistance) or reverses from R1
            if close[i] >= r3_level:
                exit_short = True
            elif close[i] > r1_level and i > 1 and close[i-1] <= r1_level:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S1 level with volume confirmation
            long_entry = (abs(close[i] - s1_level) < 0.001 * close[i]) and vol_confirmed
            
            # Short entry: price touches R1 level with volume confirmation
            short_entry = (abs(close[i] - r1_level) < 0.001 * close[i]) and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals