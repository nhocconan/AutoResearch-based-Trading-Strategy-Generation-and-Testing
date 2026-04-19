#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_R1S1_Breakout_Volume_Scalp_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Based on previous day's high, low, close
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    # Pivot point
    pp = (phigh + plow + pclose) / 3.0
    # Camarilla levels
    r1 = pp + (phigh - plow) * 1.1 / 12
    s1 = pp - (phigh - plow) * 1.1 / 12
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike filter: current 4h volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume
            if close[i] > r1_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Exit long when price returns to pivot or below S1
            if close[i] <= pp[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Exit short when price returns to pivot or above R1
            if close[i] >= pp[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals