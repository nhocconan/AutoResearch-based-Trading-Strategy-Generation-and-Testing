#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: Camarilla pivot levels from 1-day timeframe provide strong support/resistance.
Price touching S3/R3 levels with volume confirmation triggers mean-reversion trades.
Works in both bull and bear markets as it fades extremes at statistically significant levels.
Designed for 12h timeframe with ~20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
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
    
    # 1D data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # S1 = C - (H-L)*1.12/12, S2 = C - (H-L)*1.12/6, S3 = C - (H-L)*1.12/4
    # R1 = C + (H-L)*1.12/12, R2 = C + (H-L)*1.12/6, R3 = C + (H-L)*1.12/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate levels (using previous day's data)
    hl_range = high_1d - low_1d
    s3 = close_1d - hl_range * 1.12 / 4.0
    s2 = close_1d - hl_range * 1.12 / 6.0
    s1 = close_1d - hl_range * 1.12 / 12.0
    r1 = close_1d + hl_range * 1.12 / 12.0
    r2 = close_1d + hl_range * 1.12 / 6.0
    r3 = close_1d + hl_range * 1.12 / 4.0
    
    # Align to 12h timeframe (shifted by 1 for previous day's levels)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume confirmation: 24-period average (2 days of 12h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(s3_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 (take profit) or breaks below S3 (stop)
            if close[i] <= s1_12h[i] or close[i] < s3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 (take profit) or breaks above R3 (stop)
            if close[i] >= r1_12h[i] or close[i] > r3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S3 with volume confirmation (mean reversion up)
            if abs(close[i] - s3_12h[i]) < (hl_range[i-1] * 0.001 if i > 0 and not np.isnan(hl_range[i-1]) else 0.01) and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 with volume confirmation (mean reversion down)
            elif abs(close[i] - r3_12h[i]) < (hl_range[i-1] * 0.001 if i > 0 and not np.isnan(hl_range[i-1]) else 0.01) and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals