#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Volume_Filter_v1
Weekly Pivot levels (R1/S1) breakout on daily timeframe with volume spike filter.
Long when price breaks above weekly R1 with volume > 1.5x average.
Short when price breaks below weekly S1 with volume > 1.5x average.
Exit when price returns to weekly pivot (PP) level.
Designed to capture institutional breakouts with institutional volume.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Points (calculate from weekly OHLC) ===
    df_weekly = get_htf_data(prices, '1w')
    # Weekly high, low, close
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Resistance 1 (R1) = (2 * PP) - L
    r1 = (2 * pp) - weekly_low
    # Support 1 (S1) = (2 * PP) - H
    s1 = (2 * pp) - weekly_high
    
    # Align weekly levels to daily timeframe (with proper delay for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Volume filter: 1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike
            if (close[i] > r1_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly S1 with volume spike
            elif (close[i] < s1_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to weekly pivot point
        elif position == 1:
            # Exit long: price returns to or below weekly PP
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above weekly PP
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0