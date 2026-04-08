#!/usr/bin/env python3
# 12h_camarilla_pivot_breakout_volume_v2
# Hypothesis: Uses 12h Camarilla pivot levels (resistance/support) from daily data, with volume confirmation.
# Long when: price breaks above R4 with volume surge.
# Short when: price breaks below S4 with volume surge.
# Exit when price returns to the daily pivot point (midpoint) or volume drops below average.
# Designed for 12h timeframe to avoid overtrading, targeting 15-35 trades per year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # R2 = Close + 0.6 * (High - Low)
    # R1 = Close + 0.3 * (High - Low)
    # PP = (High + Low + Close) / 3
    # S1 = Close - 0.3 * (High - Low)
    # S2 = Close - 0.6 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    rng = high_1d - low_1d
    r4 = close_1d + 1.5 * rng
    r3 = close_1d + 1.1 * rng
    r2 = close_1d + 0.6 * rng
    r1 = close_1d + 0.3 * rng
    pp = (high_1d + low_1d + close_1d) / 3.0
    s1 = close_1d - 0.3 * rng
    s2 = close_1d - 0.6 * rng
    s3 = close_1d - 1.1 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align daily levels to 12h timeframe (using previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume filter: 1.5x 24-period average (approx 12 days)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period  # Wait for volume MA to be ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point or volume drops below average
            if close[i] <= pp_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point or volume drops below average
            if close[i] >= pp_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above R4 with volume surge
            if close[i] > r4_aligned[i] and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S4 with volume surge
            elif close[i] < s4_aligned[i] and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals