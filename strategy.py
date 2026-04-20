#!/usr/bin/env python3
# 12h_1d_Camarilla_R1S1_Breakout_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from 1d provide institutional support/resistance. 
# Breakout above R1 or below S1 with volume confirmation captures institutional breakouts.
# Works in bull/bear: Breakouts work in trending markets, while volume confirmation avoids false breaks in ranging markets.
# Uses 12h timeframe for execution, targeting 15-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Camarilla pivot levels on 1d (based on previous day) ===
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pp = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r1 = pp + (range_prev * 1.1 / 12)
    s1 = pp - (range_prev * 1.1 / 12)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d Camarilla data to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after we have previous day data
        # Get values
        close_val = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume confirmation
            if close_val > r1_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation
            elif close_val < s1_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals