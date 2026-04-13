#!/usr/bin/env python3
"""
6h_1d_Pivot_Reversion
Hypothesis: Fade at 1d pivot support/resistance with volume exhaustion during 6h range-bound markets.
In BTC/ETH, price often reverts to daily pivot after overextended moves, especially in low volatility.
Works in both bull (fade from R1/R2) and bear (fade from S1/S2) markets by exploiting mean reversion.
Targets 15-25 trades/year with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points and support/resistance levels."""
    P = (high + low + close) / 3.0
    R1 = 2 * P - low
    S1 = 2 * P - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    R3 = high + 2 * (P - low)
    S3 = low - 2 * (high - P)
    return P, R1, R2, R3, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d pivot points
    P_1d, R1_1d, R2_1d, R3_1d, S1_1d, S2_1d, S3_1d = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 6h timeframe
    P_1d_aligned = align_htf_to_ltf(prices, df_1d, P_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume exhaustion: current volume < 0.7 * 20-period average (low interest at extremes)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_exhaustion = volume < (vol_ma_20 * 0.7)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(P_1d_aligned[i]) or np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_exhaustion[i])):
            signals[i] = 0.0
            continue
        
        # Calculate distance from pivot as percentage of ATR
        dist_from_pivot = abs(close[i] - P_1d_aligned[i]) / atr[i]
        
        # Long: price near S1 with volume exhaustion and not too far from pivot
        long_condition = (close[i] <= S1_1d_aligned[i] * 1.001) and volume_exhaustion[i] and (dist_from_pivot < 3.0)
        
        # Short: price near R1 with volume exhaustion and not too far from pivot
        short_condition = (close[i] >= R1_1d_aligned[i] * 0.999) and volume_exhaustion[i] and (dist_from_pivot < 3.0)
        
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

name = "6h_1d_Pivot_Reversion"
timeframe = "6h"
leverage = 1.0