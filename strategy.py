#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Hypothesis: Daily breakouts at weekly Camarilla pivot levels (H3/L3) with volume confirmation (>1.5x 20-day average volume).
# Weekly Camarilla levels act as strong support/resistance; breaks signal momentum continuation.
# Volume filter reduces false breakouts. Designed to work in both bull and bear markets by capturing
# continuation moves after periods of consolidation. Target: 15-25 trades per year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly close for Camarilla levels
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/2
    camarilla_h3 = weekly_close + (weekly_high - weekly_low) * 1.1 / 2
    camarilla_l3 = weekly_close - (weekly_high - weekly_low) * 1.1 / 2
    
    # Align Camarilla levels to daily timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L3 level
            if close[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H3 level
            if close[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume confirmation
            if close[i] > camarilla_h3_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals