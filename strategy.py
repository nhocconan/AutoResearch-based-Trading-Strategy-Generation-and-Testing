#!/usr/bin/env python3
# 12h_camilla_pivot_breakout_volume_v2
# Hypothesis: Combines daily Camarilla pivot levels with 12h price breakout and volume confirmation.
# Long when price breaks above R4 level with volume > 1.5x average.
# Short when price breaks below S4 level with volume > 1.5x average.
# Exit when price returns to Pivot Point (PP) level or volume drops below average.
# Uses strict breakout conditions to limit trades (target: 15-30/year) and reduce fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camilla_pivot_breakout_volume_v2"
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
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3
    r4 = pp + (high_1d - low_1d) * 1.1 / 2
    s4 = pp - (high_1d - low_1d) * 1.1 / 2
    
    # Align daily pivot levels to 12h timeframe (previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period  # Wait for volume MA to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to PP or volume drops below average
            if close[i] <= pp_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to PP or volume drops below average
            if close[i] >= pp_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above R4 with volume surge
            if (close[i] > r4_aligned[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S4 with volume surge
            elif (close[i] < s4_aligned[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals