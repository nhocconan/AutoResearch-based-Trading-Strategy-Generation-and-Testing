#!/usr/bin/env python3
# 6h_1d_camdash_breakout_v1
# Strategy: 6h price crossing Camarilla pivot levels from 1d with volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) act as strong support/resistance.
# In trending markets, breaks above R4 or below S4 with volume continuation signal strong momentum.
# In ranging markets, reversals at R3/S3 offer mean-reversion opportunities.
# Volume filter ensures institutional participation, reducing false breaks.
# Works in bull markets via R4 breakouts and in bear markets via S4 breakdowns.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camdash_breakout_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Formula: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r4_1d = close_1d + 1.5 * range_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d
    s4_1d = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average of 6h volume
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r4_1d_aligned[i]  # Break above R4
        breakout_down = close[i] < s4_1d_aligned[i]  # Break below S4
        
        # Reversal conditions
        reversal_long = close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1]  # Cross above S3
        reversal_short = close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1]  # Cross below R3
        
        # Entry logic
        if breakout_up and vol_confirm[i] and position != 1:
            # Bullish breakout with volume
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_confirm[i] and position != -1:
            # Bearish breakdown with volume
            position = -1
            signals[i] = -0.25
        elif reversal_long and not vol_confirm[i] and position != 1:
            # Mean reversion long at S3 (only in low volume - ranging market)
            position = 1
            signals[i] = 0.25
        elif reversal_short and not vol_confirm[i] and position != -1:
            # Mean reversion short at R3 (only in low volume - ranging market)
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (close[i] < s3_1d_aligned[i] or breakout_down):
            # Exit long if price returns below S3 or bearish breakdown occurs
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > r3_1d_aligned[i] or breakout_up):
            # Exit short if price returns above R3 or bullish breakout occurs
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals