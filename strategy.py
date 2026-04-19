# 6h_1w_Pivot_R4_S4_Breakout_Volume
# Uses weekly pivot levels (R4/S4) from 1w timeframe for breakout signals
# In bull markets: breakout above R4 with volume continuation
# In bear markets: breakdown below S4 with volume continuation
# Weekly pivots provide strong support/resistance that works across market regimes
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Pivot_R4_S4_Breakout_Volume"
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
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    def calculate_pivot_points(high_arr, low_arr, close_arr):
        n_periods = len(close_arr)
        P = np.full(n_periods, np.nan)
        R1 = np.full(n_periods, np.nan)
        R2 = np.full(n_periods, np.nan)
        R3 = np.full(n_periods, np.nan)
        R4 = np.full(n_periods, np.nan)
        S1 = np.full(n_periods, np.nan)
        S2 = np.full(n_periods, np.nan)
        S3 = np.full(n_periods, np.nan)
        S4 = np.full(n_periods, np.nan)
        
        for i in range(1, n_periods):
            # Use previous week's OHLC
            high_prev = high_arr[i-1]
            low_prev = low_arr[i-1]
            close_prev = close_arr[i-1]
            
            # Standard pivot point calculation
            P[i] = (high_prev + low_prev + close_prev) / 3.0
            
            # Support and resistance levels
            R1[i] = 2.0 * P[i] - low_prev
            S1[i] = 2.0 * P[i] - high_prev
            R2[i] = P[i] + (high_prev - low_prev)
            S2[i] = P[i] - (high_prev - low_prev)
            R3[i] = high_prev + 2.0 * (P[i] - low_prev)
            S3[i] = low_prev - 2.0 * (high_prev - P[i])
            R4[i] = R3[i] + (high_prev - low_prev)
            S4[i] = S3[i] - (high_prev - low_prev)
        
        return P, R1, R2, R3, R4, S1, S2, S3, S4
    
    _, _, _, _, R4_1w, _, _, _, S4_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Align weekly pivot levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    # Volume spike detection (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Ensure enough data for volume MA (20) + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when price breaks above weekly R4 with volume confirmation
            if close[i] > R4_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below weekly S4 with volume confirmation
            elif close[i] < S4_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls back below weekly R4 (failure)
            if close[i] < R4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises back above weekly S4 (failure)
            if close[i] > S4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals