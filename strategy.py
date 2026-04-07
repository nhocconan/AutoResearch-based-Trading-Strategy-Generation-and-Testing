#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v8
Hypothesis: On 12-hour timeframe, trade reversions from daily Camarilla pivot levels with volume confirmation.
Go long when price touches or breaks below S3 level and reverses upward with volume spike.
Go short when price touches or breaks above R3 level and reverses downward with volume spike.
Exit when price reaches opposite S3/R3 level or midpoint.
Camarilla levels from daily timeframe provide institutional support/resistance that works in both trending and ranging markets.
Volume spike confirms institutional interest at these key levels.
Designed for 15-25 trades/year to minimize fee drag while capturing meaningful reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v8"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    HLC = (high_1d + low_1d + close_1d)
    range_1d = high_1d - low_1d
    
    # Levels: S1, S2, S3, S4, R1, R2, R3, R4
    # S3 = C - (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/2
    s3 = close_1d - (range_1d * 1.1 / 2)
    r3 = close_1d + (range_1d * 1.1 / 2)
    s4 = close_1d - (range_1d * 1.1)
    r4 = close_1d + (range_1d * 1.1)
    
    # Align to 12h timeframe (previous day's levels for current day)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume filter: 24-period average (2 days of 12h data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current volume > 1.8x average
        vol_ok = volume[i] > 1.8 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S4 or R3 (opposite extreme or resistance)
            if low[i] <= s4_aligned[i] or high[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 or S3 (opposite extreme or support)
            if high[i] >= r4_aligned[i] or low[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price touches/below S3 and shows reversal (close > low)
                if low[i] <= s3_aligned[i] and close[i] > low[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: price touches/above R3 and shows reversal (close < high)
                elif high[i] >= r3_aligned[i] and close[i] < high[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals