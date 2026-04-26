#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA34 trend filter and volume confirmation. 
Only long when price > weekly EMA34, short when price < weekly EMA34. Uses fixed position size 0.25 
to minimize fee churn. Targets 50-100 total trades over 4 years (12-25/year). 
Designed to capture breakouts aligned with weekly trend while filtering counter-trend noise.
Works in both bull (breakouts with trend) and bear (breakouts against trend filtered by weekly EMA) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous daily data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, S1 (using inner 3/8 and 1/4 levels)
    camarilla_multiplier = 1.1 / 4
    r1 = close_1d + range_1d * camarilla_multiplier
    s1 = close_1d - range_1d * camarilla_multiplier
    pp = typical_price  # pivot point
    
    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Load 1-week data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0 * 50-period EMA volume (daily)
    avg_volume = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long logic: price breaks above R1 with volume spike and above weekly EMA34
        if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_34_1w_aligned[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S1 with volume spike and below weekly EMA34
        elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_34_1w_aligned[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to pivot point
        elif position == 1 and close[i] < pp_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > pp_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0