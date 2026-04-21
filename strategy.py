#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirm
Hypothesis: Use daily Camarilla pivot levels (R1/S1) from 1d timeframe. 
Long when price breaks above R1 with volume confirmation. 
Short when price breaks below S1 with volume confirmation.
Exit when price crosses the pivot point (PP) or reverses.
Works in bull markets by buying breakouts above R1 and in bear markets by selling breakdowns below S1.
Volume confirmation filters false breakouts. Pivot levels provide institutional support/resistance.
Designed for 4h timeframe to capture medium-term moves with ~20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (using previous day's data)
    pp = np.full_like(high_1d, np.nan)
    r1 = np.full_like(high_1d, np.nan)
    s1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        # Use previous day's OHLC to calculate today's pivot levels
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        pp[i] = (phigh + plow + pclose) / 3
        r1[i] = pp[i] + (phigh - plow) * 1.1 / 12
        s1[i] = pp[i] - (phigh - plow) * 1.1 / 12
    
    # Align pivots to 4h timeframe (using previous day's levels for current day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if pivots not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < s1_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point or loses momentum
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point or loses momentum
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0