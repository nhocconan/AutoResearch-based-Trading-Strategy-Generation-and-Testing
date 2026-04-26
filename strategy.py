#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_v1
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d EMA34 trend filter and volume confirmation. 
Only long when price > EMA34(1d), short when price < EMA34(1d). Uses fixed position size (0.25) 
to minimize fee churn. Designed for 12h timeframe targeting 50-150 trades over 4 years 
(12-37/year) with strong performance in both bull and bear regimes via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, S1, PP
    camarilla_multiplier = 1.1 / 4
    r1 = close_1d + range_1d * camarilla_multiplier
    pp = typical_price
    s1 = close_1d - range_1d * camarilla_multiplier
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 1d data for EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Fixed position size to minimize fee churn
        base_size = 0.25
        
        # Long logic: price breaks above R1 with volume spike and above 1d EMA34
        if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 with volume spike and below 1d EMA34
        elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to pivot point or opposite breakout
        elif position == 1 and (close[i] < pp_aligned[i] or close[i] < s1_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > pp_aligned[i] or close[i] > r1_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_v1"
timeframe = "12h"
leverage = 1.0