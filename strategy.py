#!/usr/bin/env python3
"""
12h_1w_camarilla_breakout_volume_regime
Hypothesis: Weekly Camarilla levels on 12h chart with volume confirmation and chop regime filter.
Works in bull/bear by using weekly structure (more robust than daily) and volume to confirm breakouts.
Chop filter avoids whipsaws in ranging markets. Target: 15-30 trades/year (60-120 total over 4 years).
"""

name = "12h_1w_camarilla_breakout_volume_regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for more robust structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's range
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Weekly Camarilla levels
    range_ = prev_high - prev_low
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Chop regime filter (14-period)
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(close_1w, 1))
    low_close = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    true_range = pd.Series(tr)
    chop = 100 * np.log10(atr_sum / true_range.rolling(window=14, min_periods=14).sum()) / np.log10(14)
    chop = chop.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Align weekly levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low chop (trending market)
        if chop_aligned[i] > 61.8:  # choppy range
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long: break above weekly R4 with volume
        if close[i] > r4_aligned[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: break below weekly S4 with volume
        elif close[i] < s4_aligned[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals