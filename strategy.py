#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pullback_Reversal
Hypothesis: Trade reversions from Camarilla pivot levels on 12h timeframe with 1d trend filter and volume confirmation.
In ranging markets (2025-2026), price often reverts to mean after touching S3/R3 levels.
In trending markets, pullbacks to S4/R4 with 1d trend alignment offer high-probability entries.
Volume spike confirms institutional interest at these key levels.
Target: 15-25 trades/year to minimize fee drag while capturing mean reversion and trend pullbacks.
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Typical = (H+L+C)/3
    typical = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    S1 = close_1d - (range_hl * 1.1 / 12)
    S2 = close_1d - (range_hl * 1.1 / 6)
    S3 = close_1d - (range_hl * 1.1 / 4)
    S4 = close_1d - (range_hl * 1.1 / 2)
    R1 = close_1d + (range_hl * 1.1 / 12)
    R2 = close_1d + (range_hl * 1.1 / 6)
    R3 = close_1d + (range_hl * 1.1 / 4)
    R4 = close_1d + (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(S3_12h[i]) or np.isnan(R3_12h[i]) or np.isnan(S4_12h[i]) or 
            np.isnan(R4_12h[i]) or np.isnan(ema_50_12h[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: price touches S3/S4 with volume expansion and above daily EMA50 (bullish bias)
        long_condition = volume_expansion[i] and (close[i] <= S3_12h[i] or close[i] <= S4_12h[i]) and (close[i] > ema_50_12h[i])
        
        # Short: price touches R3/R4 with volume expansion and below daily EMA50 (bearish bias)
        short_condition = volume_expansion[i] and (close[i] >= R3_12h[i] or close[i] >= R4_12h[i]) and (close[i] < ema_50_12h[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Camarilla_Pullback_Reversal"
timeframe = "12h"
leverage = 1.0