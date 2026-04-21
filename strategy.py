#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Camarilla pivot levels from 1d timeframe with volume confirmation.
Camarilla levels act as support/resistance where price often reverses or accelerates.
Breakouts above R3 or below S3 with volume confirmation indicate strong momentum.
Works in both bull and bear markets by capturing breakout moves.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500), etc.
    # S4 = C - ((H-L) * 1.5000), S3 = C - ((H-L) * 1.2500), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ranges
    rng = high_1d - low_1d
    
    # Camarilla levels
    R3 = close_1d + (rng * 1.2500)
    S3 = close_1d - (rng * 1.2500)
    R4 = close_1d + (rng * 1.5000)
    S4 = close_1d - (rng * 1.5000)
    
    # Align to 4h timeframe (1 day = 6 * 4h bars)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: current volume vs 24-period average (24*4h = 4 days)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=24, min_periods=24).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if indicators not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 2.0  # Require strong volume confirmation
        
        if position == 0:
            # Enter long: price breaks above R3 with volume
            if (price_close > R3_aligned[i] and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume
            elif (price_close < S3_aligned[i] and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite S3/R3 level or volume dries up
            if position == 1 and (price_close < S3_aligned[i] or vol_ratio_val < 1.0):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > R3_aligned[i] or vol_ratio_val < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_Breakout_Volume"
timeframe = "4h"
leverage = 1.0