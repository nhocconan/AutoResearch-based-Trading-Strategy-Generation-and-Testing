#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Pivot_R4S4_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_12h + low_12h + close_12h) / 3
    
    # Calculate pivot and support/resistance levels
    pivot = typical_price
    range_hl = high_12h - low_12h
    
    # Camarilla levels
    R4 = close_12h + range_hl * 1.1 / 2
    R3 = close_12h + range_hl * 1.1 / 4
    R2 = close_12h + range_hl * 1.1 / 6
    R1 = close_12h + range_hl * 1.1 / 12
    S1 = close_12h - range_hl * 1.1 / 12
    S2 = close_12h - range_hl * 1.1 / 6
    S3 = close_12h - range_hl * 1.1 / 4
    S4 = close_12h - range_hl * 1.1 / 2
    
    # Align levels to 6h timeframe
    R4_6h = align_htf_to_ltf(prices, df_12h, R4)
    R3_6h = align_htf_to_ltf(prices, df_12h, R3)
    S3_6h = align_htf_to_ltf(prices, df_12h, S3)
    S4_6h = align_htf_to_ltf(prices, df_12h, S4)
    
    # Volume spike filter: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long breakout above R4 with volume spike
            if close[i] > R4_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakdown below S4 with volume spike
            elif close[i] < S4_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below R3 (take profit) or reverses below S4 (stop)
            if close[i] < R3_6h[i]:
                signals[i] = 0.0  # Take profit at R3
                position = 0
            elif close[i] < S4_6h[i]:
                signals[i] = 0.0  # Stop loss at S4
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above S3 (take profit) or reverses above R4 (stop)
            if close[i] > S3_6h[i]:
                signals[i] = 0.0  # Take profit at S3
                position = 0
            elif close[i] > R4_6h[i]:
                signals[i] = 0.0  # Stop loss at R4
                position = 0
            else:
                signals[i] = -0.25
    
    return signals