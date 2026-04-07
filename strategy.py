#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v1
Hypothesis: On 4-hour timeframe, use daily Camarilla pivot levels with volume confirmation.
Long when price breaks above R4 level with volume > 1.5x 20-period average.
Short when price breaks below S4 level with volume > 1.5x 20-period average.
Exit when price returns to the Pivot Point (PP) level.
Designed for 20-50 trades/year to minimize fee drag while capturing institutional breakout moves.
Works in both bull and bear markets as Camarilla levels adapt to volatility and volume filter ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot point and ranges
    pp = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r4 = pp + range_val * 1.1 / 2
    r3 = pp + range_val * 1.1 / 4
    r2 = pp + range_val * 1.1 / 6
    r1 = pp + range_val * 1.1 / 12
    s1 = pp - range_val * 1.1 / 12
    s2 = pp - range_val * 1.1 / 6
    s3 = pp - range_val * 1.1 / 4
    s4 = pp - range_val * 1.1 / 2
    
    # Align to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to pivot point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price breaks above R4
                if close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S4
                elif close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals