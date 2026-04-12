#!/usr/bin/env python3
"""
4h_12h_keltner_reversion
Uses Keltner Channel (ATR-based) on 12h to detect overextended moves.
When price touches upper/lower Keltner band and shows exhaustion (volume declining),
we take counter-trend position expecting reversion to mean (middle band).
Works in ranging markets and during pullbacks in trends.
"""

name = "4h_12h_keltner_reversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Keltner Channel calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Keltner Channel (20, 1.5) on 12h
    kc_length = 20
    kc_mult = 1.5
    
    # Middle line (EMA)
    midline = pd.Series(close_12h).ewm(span=kc_length, adjust=False, min_periods=kc_length).mean().values
    
    # Average True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=kc_length, adjust=False, min_periods=kc_length).mean().values
    
    # Upper and lower bands
    upper = midline + (kc_mult * atr)
    lower = midline - (kc_mult * atr)
    
    # Volume exhaustion: current volume < 80% of 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_exhaust = volume < (vol_ma * 0.8)
    
    # Align Keltner Channel to 4h
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    midline_aligned = align_htf_to_ltf(prices, df_12h, midline)
    vol_exhaust_aligned = align_htf_to_ltf(prices, df_12h, vol_exhaust)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midline_aligned[i]) or np.isnan(vol_exhaust_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches lower Keltner band with volume exhaustion
        if close[i] <= lower_aligned[i] and vol_exhaust_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price touches upper Keltner band with volume exhaustion
        elif close[i] >= upper_aligned[i] and vol_exhaust_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to midline
        elif position == 1 and close[i] >= midline_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= midline_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals