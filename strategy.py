#!/usr/bin/env python3
name = "6h_12h_Camarilla_Pivot_Reversal"
timeframe = "6h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels using previous 12h bar
    # H, L, C from previous 12h bar
    h_prev = df_12h['high'].shift(1).values
    l_prev = df_12h['low'].shift(1).values
    c_prev = df_12h['close'].shift(1).values
    
    # Calculate pivot and levels
    pp = (h_prev + l_prev + c_prev) / 3
    range_hl = h_prev - l_prev
    r3 = c_prev + range_hl * 1.1 / 2
    s3 = c_prev - range_hl * 1.1 / 2
    r4 = c_prev + range_hl * 1.1
    s4 = c_prev - range_hl * 1.1
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: 12-period average (3 days of 6h bars)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 12  # Wait for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S3 with volume, targeting S4 as stop
            vol_condition = volume[i] > vol_ma_12[i] * 1.8
            
            if close[i] > s3_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R3 with volume, targeting R4 as stop
            elif close[i] < r3_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches S4 (target) or closes back below S3 (failed breakout)
            if close[i] >= s4_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches R4 (target) or closes back above R3 (failed breakdown)
            if close[i] <= r4_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Camarilla pivot reversal with 12h structure
# - Camarilla R3/S3 act as intraday support/resistance levels where reversals often occur
# - Break of R3/S3 with volume indicates potential continuation to R4/S4 targets
# - Fade at R3/S3 in ranging markets, breakout continuation in trending markets
# - Volume confirmation (1.8x average) filters false breakouts
# - Works in both bull (buy S3 breaks in uptrends) and bear (sell R3 breaks in downtrends)
# - Target R4/S4 levels provide clear profit targets
# - Position size 0.25 targets 50-150 trades over 4 years, avoiding fee drag
# - Uses 12h timeframe for structure, 6h for execution timing