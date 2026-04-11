#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot breakout with 12h volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (derived from 12h high/low/close) act as strong support/resistance.
# Breakouts above/below these levels with volume > 1.5x 20-period average capture momentum.
# Designed for low trade frequency (<25/year) to minimize fee drift.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from 12h data
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    r4_12h = c_12h + ((h_12h - l_12h) * 1.1 / 2)
    r3_12h = c_12h + ((h_12h - l_12h) * 1.1 / 4)
    r2_12h = c_12h + ((h_12h - l_12h) * 1.1 / 6)
    r1_12h = c_12h + ((h_12h - l_12h) * 1.1 / 12)
    pp_12h = (h_12h + l_12h + c_12h) / 3
    s1_12h = c_12h - ((h_12h - l_12h) * 1.1 / 12)
    s2_12h = c_12h - ((h_12h - l_12h) * 1.1 / 6)
    s3_12h = c_12h - ((h_12h - l_12h) * 1.1 / 4)
    s4_12h = c_12h - ((h_12h - l_12h) * 1.1 / 2)
    
    # Align Camarilla levels to 4h
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # 12h volume average (20-period) for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Align raw 12h volume for confirmation
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or np.isnan(r2_12h_aligned[i]) or \
           np.isnan(r1_12h_aligned[i]) or np.isnan(pp_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or \
           np.isnan(s2_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or \
           np.isnan(vol_avg_20_12h_aligned[i]) or np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_confirm = vol_12h_aligned[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Entry conditions
        # Long: Price crosses above R3 level AND volume confirmation
        if close[i] > r3_12h_aligned[i] and vol_confirm and position != 1:
            # Additional check: ensure we didn't just cross above R3 in previous bar
            if i == 50 or close[i-1] <= r3_12h_aligned[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price crosses below S3 level AND volume confirmation
        elif close[i] < s3_12h_aligned[i] and vol_confirm and position != -1:
            # Additional check: ensure we didn't just cross below S3 in previous bar
            if i == 50 or close[i-1] >= s3_12h_aligned[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price returns to pivot point (mean reversion)
        elif position == 1 and close[i] < pp_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pp_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals